import argparse
import shutil
import torch
import time
from datetime import datetime
import os
from tqdm import tqdm
import yaml
import pytz
import numpy as np
import json

from accelerate.utils import set_seed, broadcast_object_list
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import FluxPipeline
from torch.utils.tensorboard import SummaryWriter
from diffusers.optimization import get_scheduler
from torchvision.utils import make_grid
from diffusers.utils.torch_utils import is_compiled_module
from peft.utils import get_peft_model_state_dict

from models.unifined_model import UniModel, load_lora_weights_into_transformer
from utils.others import cycle
from dataset import LMDBDataset


def prepare_images_for_saving(image_tensor):
    grid_image = make_grid(image_tensor.cpu(), nrow=int(np.sqrt(image_tensor.size(0))), padding=2)
    image = grid_image.permute(1, 2, 0).mul(255).byte().numpy()
    return image


class Trainer:
    def __init__(self, args):
        self.args = args

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        accelerator_project_config = ProjectConfiguration(logging_dir=args.log_path)
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision="no",
            step_scheduler_with_optimizer=False,
            log_with="tensorboard",
            project_config=accelerator_project_config,
            kwargs_handlers=None,
        )
        if args.checkpoint_path:
            saved_config, _ = read_resume_config(args.checkpoint_path)
            saved_world_size = saved_config.get("world_size")
            if saved_world_size is not None and saved_world_size != accelerator.num_processes:
                raise ValueError(
                    f"Resume requires {saved_world_size} processes, got {accelerator.num_processes}. "
                    "Use --lora_ckpt for a weights-only start with a different batch configuration."
                )
        seed = [args.seed]
        if accelerator.is_main_process and seed[0] is None:
            rng = torch.Generator(device="cpu")
            seed[0] = rng.seed() % (2 ** 31)
        broadcast_object_list(seed)
        args.seed = seed[0]
        set_seed(args.seed + accelerator.process_index)

        print(accelerator.state)

        run_name = [None]
        if accelerator.is_main_process:
            run_name[0] = datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d_%H-%M-%S")
        broadcast_object_list(run_name)
        current_time = run_name[0]
        self.output_path = os.path.join(args.output_path, f"{current_time}_seed{args.seed}")

        if accelerator.is_main_process:
            output_path = self.output_path
            os.makedirs(output_path, exist_ok=False)

            args_dict = vars(args).copy()
            args_dict['world_size'] = accelerator.num_processes
            args_dict['global_batch_size'] = args.batch_size * accelerator.num_processes
            args_dict['adl_definition'] = 'Mean image-token-to-mean cosine similarity over all transformer layers'
            args_file_path = os.path.join(output_path, 'config.yaml')
            with open(args_file_path, 'w') as yaml_file:
                yaml.dump(args_dict, yaml_file, default_flow_style=False)

            os.makedirs(args.log_path, exist_ok=True)

            experiment_name = os.path.basename(args.output_path) + '/' + os.path.basename(output_path)
            log_dir = os.path.join(args.log_path, experiment_name)
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard log directory: {log_dir}")
            os.makedirs(log_dir, exist_ok=True)

        # Fixed student text conditions are distributed in weights/fluxsr/.
        prompt_embeds = torch.load(args.t5_embd_path, map_location='cpu', weights_only=True)
        pooled_prompt_embeds = torch.load(args.clip_embd_path, map_location='cpu', weights_only=True)
        if prompt_embeds.shape[0] == 1 and args.batch_size > 1:
            prompt_embeds = prompt_embeds.repeat(args.batch_size, 1, 1)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(args.batch_size, 1)
        self.t5_embedding = prompt_embeds.to(accelerator.device)
        self.clip_embedding = pooled_prompt_embeds.to(accelerator.device)

        self.model = UniModel(args, accelerator)

        dataset = LMDBDataset(args)

        self.dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=args.num_workers,
            batch_size=args.batch_size, shuffle=True,
            drop_last=True, pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

        self.max_grad_norm = args.max_grad_norm
        self.step = 0

        self.optimizer_generator = torch.optim.AdamW(
            [param for param in self.model.feedforward_model.parameters() if param.requires_grad],
            lr=args.lr,
            betas=(0.9, 0.999),  # pytorch's default
            weight_decay=0.01  # pytorch's default
        )

        generator_params = sum(p.numel() for p in self.model.feedforward_model.parameters() if p.requires_grad)
        print(f"Generator Optimizer is optimizing {generator_params} parameters.")

        self.scheduler_generator = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer_generator,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters
        )

        (
            self.model.feedforward_model, self.optimizer_generator, self.scheduler_generator, self.dataloader
        ) = accelerator.prepare(
            self.model.feedforward_model, self.optimizer_generator, self.scheduler_generator, self.dataloader
        )

        if len(self.dataloader) == 0:
            raise ValueError("No full training batch is available; reduce --batch_size or add teacher pairs.")
        self.dataloader = cycle(self.dataloader)

        self.accelerator = accelerator
        # Save trainable adapters and optimizer/RNG state.
        accelerator.register_save_state_pre_hook(self.save_lora_hook)
        accelerator.register_load_state_pre_hook(self.load_lora_hook)

        self.train_iters = args.train_iters
        self.batch_size = args.batch_size
        self.log_iters = args.log_iters
        self.log_loss = args.log_loss

        self.max_checkpoint = args.max_checkpoint

        self.previous_time = None
        self.no_save = args.no_save

        if args.checkpoint_path is not None:
            self.load(args.checkpoint_path)

    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save(self):
        output_path = os.path.join(self.output_path, f"checkpoint_{self.step:06d}")
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(output_path)
        if self.accelerator.is_main_process:
            with open(os.path.join(output_path, 'training_config.json'), 'w') as f:
                json.dump(dict(vars(self.args), completed_steps=self.step,
                               world_size=self.accelerator.num_processes,
                               global_batch_size=self.batch_size * self.accelerator.num_processes),
                          f, indent=2)
            checkpoints = sorted(
                (int(folder.removeprefix("checkpoint_")), folder)
                for folder in os.listdir(self.output_path)
                if folder.startswith("checkpoint_")
                and folder.removeprefix("checkpoint_").isdigit()
                and os.path.isdir(os.path.join(self.output_path, folder))
            )
            keep_every = self.args.checkpoint_keep_every
            # Milestone checkpoints never enter the rolling deletion list.
            rolling = [folder for step, folder in checkpoints
                       if not (keep_every > 0 and step > 0 and step % keep_every == 0)]
            for folder in rolling[:max(0, len(rolling) - self.max_checkpoint)]:
                shutil.rmtree(os.path.join(self.output_path, folder))
            print(f"Saved checkpoint: {output_path}", flush=True)
        self.accelerator.wait_for_everyone()

    def save_lora_hook(self, models, weights, output_dir):
        if self.accelerator.is_main_process:
            transformer = self.unwrap_model(self.model.feedforward_model)
            FluxPipeline.save_lora_weights(
                save_directory=output_dir,
                transformer_lora_layers=get_peft_model_state_dict(transformer),
                safe_serialization=True,
            )
        weights.clear()

    def load_lora_hook(self, models, input_dir):
        transformer = self.unwrap_model(self.model.feedforward_model)
        load_lora_weights_into_transformer(transformer, input_dir)
        models.clear()

    def load(self, checkpoint_path):
        # Accelerate restores adapters, optimizer, scheduler and per-rank RNG.
        # The shuffled data iterator restarts on resume.
        _, self.step = read_resume_config(checkpoint_path)
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed {checkpoint_path} after {self.step} completed updates")

    def train_one_step(self):
        self.model.train()
        self.model.vae.eval()
        if hasattr(self.model, 'precep_loss'):
            self.model.precep_loss.eval()
        started = time.monotonic()

        accelerator = self.accelerator

        visual = self.args.visual_iters > 0 and self.step % self.args.visual_iters == 0

        batch = next(self.dataloader)
        gt = batch['hr_images']
        lq = batch['lr_images']
        noise = batch['input_latents']
        gt_latent = batch['output_latents']
        real_train_dict = {
            "gt_image": gt,
            "gt_latent": gt_latent,
            "noise_latent": noise,
        }

        generator_loss_dict, generator_log_dict = self.model(
            lq, self.t5_embedding, self.clip_embedding,
            visual=visual,
            real_train_dict=real_train_dict,
        )

        generator_loss = 0.0

        if self.args.spatial_loss:
            generator_loss += generator_loss_dict["loss_spatial"]

        if self.args.adl_weight > 0:
            generator_loss += generator_loss_dict["loss_adl"] * self.args.adl_weight

        generator_loss += generator_loss_dict["loss_fm"]

        if not torch.isfinite(generator_loss).all():
            raise FloatingPointError(f"Non-finite training loss at step {self.step}")
        self.accelerator.backward(generator_loss)
        generator_grad_norm = accelerator.clip_grad_norm_(self.model.feedforward_model.parameters(), self.max_grad_norm)
        if not torch.isfinite(generator_grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm at step {self.step}")
        self.optimizer_generator.step()
        self.optimizer_generator.zero_grad()

        self.scheduler_generator.step()

        loss_dict = generator_loss_dict
        log_dict = generator_log_dict

        if accelerator.is_main_process and self.log_loss and (not visual):
            self.writer.add_scalar("grad_norm/generator_grad_norm", generator_grad_norm.item(), self.step)

            self.writer.add_scalar("loss/loss_fm", loss_dict['loss_fm'].item(), self.step)

            if self.args.spatial_loss:
                self.writer.add_scalar("loss/loss_spatial", loss_dict['loss_spatial'].item(), self.step)
                self.writer.add_scalar(f"loss/loss_{self.args.spatial_loss_type}", loss_dict['loss_mse'].item(), self.step)
                if self.args.percep_weight > 0:
                    self.writer.add_scalar(f"loss/loss_{self.args.percep_loss}", loss_dict['loss_percep'].item(), self.step)
                    if self.args.edge_weight > 0:
                        self.writer.add_scalar("loss/loss_edge", loss_dict['loss_edge'].item(), self.step)
                    if self.args.tv_weight > 0:
                        self.writer.add_scalar("loss/loss_tv", loss_dict['loss_tv'].item(), self.step)

            if self.args.adl_weight > 0:
                self.writer.add_scalar("loss/loss_adl", loss_dict['loss_adl'].item(), self.step)

        if visual:
            log_dict['lq_image'] = accelerator.gather(lq)
            log_dict['pred_image'] = accelerator.gather(log_dict['pred_image'])
            log_dict['gt_image'] = accelerator.gather(real_train_dict['gt_image'])

        if accelerator.is_main_process and visual:
            with torch.no_grad():
                pred_image = log_dict['pred_image']
                pred_image_grid = prepare_images_for_saving(pred_image)
                self.writer.add_images("pred_image", pred_image_grid, self.step, dataformats="HWC")

                gt_image = log_dict['gt_image']
                gt_image_grid = prepare_images_for_saving(gt_image)
                self.writer.add_images("gt_image", gt_image_grid, self.step, dataformats="HWC")

                lq_image = log_dict['lq_image']
                lq_image_grid = prepare_images_for_saving(lq_image)
                self.writer.add_images("lq_image", lq_image_grid, self.step, dataformats="HWC")

        if self.step == 0 or (self.step + 1) % self.args.status_iters == 0:
            names = [name for name, value in loss_dict.items() if torch.is_tensor(value) and value.numel() == 1]
            values = torch.stack([loss_dict[name].detach().float().reshape(()) for name in names])
            values = accelerator.reduce(values, reduction='mean').cpu().tolist()
            if accelerator.is_main_process:
                status = dict(step=self.step + 1, seconds=time.monotonic() - started,
                              lr=self.scheduler_generator.get_last_lr()[0],
                              grad_norm=float(generator_grad_norm),
                              **dict(zip(names, values)))
                print(json.dumps(status), flush=True)
                with open(os.path.join(self.output_path, 'metrics.jsonl'), 'a') as f:
                    f.write(json.dumps(status) + '\n')

    def checkpoint_due(self):
        keep_every = self.args.checkpoint_keep_every
        return self.step > 0 and (
            self.step % self.log_iters == 0
            or (keep_every > 0 and self.step % keep_every == 0)
        )

    def train(self):
        progress_bar = tqdm(range(self.step, self.train_iters), desc="Training", unit="step",
                            disable=not self.accelerator.is_main_process)
        for index in progress_bar:
            self.train_one_step()
            self.step += 1
            if (not self.no_save) and self.checkpoint_due():
                self.save()

            if self.accelerator.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    self.writer.add_scalar("time/per_iteration", current_time - self.previous_time, self.step)
                    self.previous_time = current_time

        if not self.no_save and not self.checkpoint_due():
            self.save()
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            torch.cuda.empty_cache()
            self.writer.close()

        self.accelerator.end_training()


def read_resume_config(checkpoint_path):
    """Read step/recipe metadata from a complete Accelerate training checkpoint."""
    if not os.path.isfile(os.path.join(checkpoint_path, "optimizer.bin")):
        raise ValueError(
            "--checkpoint_path requires a full training checkpoint including optimizer.bin. "
            "Use --lora_ckpt for the published weights-only adapter."
        )
    config_path = os.path.join(checkpoint_path, "training_config.json")
    config = {}
    if os.path.isfile(config_path):
        with open(config_path) as handle:
            config = json.load(handle)
    step = config.get("completed_steps")
    if step is None:
        # Compatibility with earlier checkpoint_XXXXXX directories.
        name = os.path.basename(os.path.normpath(checkpoint_path))
        if not name.startswith("checkpoint_") or not name.removeprefix("checkpoint_").isdigit():
            raise ValueError("Resume step is missing from training_config.json and checkpoint directory name")
        step = int(name.removeprefix("checkpoint_"))
    if not isinstance(step, int) or step < 0:
        raise ValueError("Checkpoint completed_steps must be a nonnegative integer")
    return config, step


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Train FluxSR LoRA from stored FLUX teacher pairs.")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument('--t5_embd_path', type=str, default='weights/fluxsr/t5_embedding.pth')
    parser.add_argument('--clip_embd_path', type=str, default='weights/fluxsr/clip_embedding.pth')
    parser.add_argument("--output_path", type=str, default="exp/fluxsr")
    parser.add_argument("--log_path", type=str, default="tb-log")
    parser.add_argument("--train_iters", type=int, default=20000, help="Total completed optimizer updates, including resumed steps")
    parser.add_argument("--log_iters", type=int, default=1000, help="Save a training checkpoint every N completed updates")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Resume adapter, optimizer, scheduler, RNG and step from a full training checkpoint")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--visual_iters", type=int, default=1000, help="TensorBoard image interval; 0 disables image logging")
    parser.add_argument("--status_iters", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help="max grad norm for network")
    parser.add_argument("--warmup_step", type=int, default=500, help="warmup step for network")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--use_fp16", action=argparse.BooleanOptionalAction, default=True,
                        help="Legacy name: use BF16 autocast and VAE; trainable LoRA parameters stay FP32")
    parser.add_argument("--enable_xformers", action="store_true")
    parser.add_argument("--no_save", action="store_true", help="don't save ckpt for debugging only")
    parser.add_argument("--log_loss", action=argparse.BooleanOptionalAction, default=True, help="Log loss to TensorBoard")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_checkpoint", type=int, default=2, help="Number of rolling checkpoints in addition to permanent milestones")
    parser.add_argument("--checkpoint_keep_every", type=int, default=5000,
                        help="Permanently retain every N completed steps; 0 disables milestones")
    parser.add_argument("--spatial_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spatial_loss_type", choices=("mse", "l1"), default="mse")
    parser.add_argument("--percep_loss", type=str, default="lpips")
    parser.add_argument("--percep_weight", type=float, default=1.0)
    parser.add_argument("--edge_weight", type=float, default=0)
    parser.add_argument("--tv_weight", type=float, default=1.0)
    parser.add_argument("--adl_weight", type=float, default=0.1)
    parser.add_argument("--conditioning_timestep", type=int, default=350)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True,
                        help="Checkpoint each FLUX block, including its scalar ADL reduction")

    parser.add_argument("--use_flux_generated_data", action="store_true", default=True,
                        help="Compatibility flag; this trainer always uses stored FLUX teacher pairs")
    parser.add_argument("--deg_file_path", type=str, default="realesrgan_paper.yaml")
    parser.add_argument("--lmdb_path", type=str, default=None)


    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=float, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_ckpt", type=str, default=None,
                        help="Weights-only warm start from a FluxSR checkpoint directory or .safetensors file; starts at step 0")

    parser.add_argument("--scale_variance", action="store_true")
    parser.add_argument("--dynamic_variance_scale_factor", action="store_true")
    parser.add_argument("--variance_scale_factor", type=float, default=1.0)

    args = parser.parse_args(input_args)
    if args.checkpoint_keep_every < 0 or args.max_checkpoint < 1 or args.log_iters < 1:
        parser.error("checkpoint_keep_every must be >= 0; max_checkpoint and log_iters must be >= 1")
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.gradient_accumulation_steps != 1:
        parser.error("Only --gradient_accumulation_steps 1 is supported")
    if not args.use_fp16:
        parser.error("The frozen FLUX base uses BF16; keep --use_fp16 enabled (its legacy name means BF16)")
    if not args.lmdb_path:
        parser.error("--lmdb_path is required (one LMDB or a directory of rank*.lmdb shards)")
    if not 0 < args.conditioning_timestep < 1000:
        parser.error("--conditioning_timestep must be between 1 and 999")
    if args.train_iters < 1 or args.batch_size < 1 or args.num_workers < 0 or args.status_iters < 1:
        parser.error("train_iters, batch_size and status_iters must be positive; num_workers must be >= 0")
    if args.lora_rank < 1 or args.lora_alpha <= 0 or not 0 <= args.lora_dropout < 1:
        parser.error("LoRA rank/alpha must be positive and dropout must be in [0, 1)")
    if min(args.percep_weight, args.tv_weight, args.adl_weight, args.edge_weight) < 0:
        parser.error("Loss weights must be nonnegative")
    if args.percep_weight == 0 and (args.tv_weight > 0 or args.edge_weight > 0):
        parser.error("TV/edge LPIPS currently require --percep_weight > 0; set their weights to 0 to disable")
    if args.lora_ckpt and args.checkpoint_path:
        parser.error("Choose --checkpoint_path for full resume or --lora_ckpt for a weights-only start")
    if args.checkpoint_path:
        try:
            saved, completed_steps = read_resume_config(args.checkpoint_path)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        # Paths, logging frequency and the new stopping step may change on resume.
        recipe_keys = (
            "batch_size", "seed", "lr", "warmup_step", "lora_rank", "lora_alpha", "lora_dropout",
            "conditioning_timestep", "guidance", "spatial_loss", "spatial_loss_type",
            "percep_loss", "percep_weight", "tv_weight", "edge_weight", "adl_weight",
            "scale_variance", "dynamic_variance_scale_factor", "variance_scale_factor",
        )
        mismatches = [f"--{key}={getattr(args, key)!r} (checkpoint: {saved[key]!r})"
                      for key in recipe_keys if key in saved and getattr(args, key) != saved[key]]
        if mismatches:
            parser.error("Resume configuration differs: " + "; ".join(mismatches))
        if completed_steps >= args.train_iters:
            parser.error(f"Checkpoint already completed {completed_steps} updates; --train_iters must be larger")

    return args


if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(args)

    trainer.train()
