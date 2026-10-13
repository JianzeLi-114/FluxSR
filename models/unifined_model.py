import copy
import math
import os
import types

import torch
import torch.nn.functional as F
from torch import nn

import pyiqa
from einops import rearrange

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.training_utils import compute_density_for_timestep_sampling
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file

from models.flux_forward import transformer_forward
from utils.others import EdgeDetectionModel, NoOpContext


def load_lora_weights_into_transformer(transformer, checkpoint_path):
    """Load a FluxSR adapter in place, before optimizer/DDP construction.

    Accepts a checkpoint directory or its ``pytorch_lora_weights.safetensors``
    file. Rank and target modules must match the adapter already on the model.
    This is a weights-only warm start, not an optimizer/RNG/step resume.
    """
    path = os.fspath(checkpoint_path)
    if os.path.isdir(path):
        path = os.path.join(path, "pytorch_lora_weights.safetensors")
    state = load_file(path, device="cpu")
    state = {key.removeprefix("transformer."): value for key, value in state.items()}
    expected = get_peft_model_state_dict(transformer, adapter_name="default")
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    if missing or unexpected:
        raise ValueError(
            "LoRA checkpoint does not match the FluxSR adapter: "
            f"missing keys={sorted(missing)}, unexpected keys={sorted(unexpected)}"
        )
    set_peft_model_state_dict(transformer, state, adapter_name="default")


class UniModel(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()

        self.args = args
        self.accelerator = accelerator
        self.conditioning_timestep = args.conditioning_timestep
        self.use_fp16 = args.use_fp16
        self.gradient_checkpointing = args.gradient_checkpointing

        # Offline teacher pairs provide epsilon - z0 directly, so training
        # only needs one FLUX transformer per worker.

        self.feedforward_model = FluxTransformer2DModel.from_pretrained(
            args.model_id, subfolder="transformer",
            torch_dtype=torch.bfloat16
        )

        self.feedforward_model.requires_grad_(False)
        lora_target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
        lora_config = LoraConfig(
            r=args.lora_rank,
            target_modules=lora_target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        self.feedforward_model.add_adapter(lora_config)
        # Preserve small AdamW updates and FP32 warm-start values. The frozen
        # FLUX base remains BF16; autocast handles the mixed-precision forward.
        for parameter in self.feedforward_model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()

        if args.lora_ckpt is not None:
            accelerator.print(f"Warm-starting LoRA weights from {args.lora_ckpt}")
            load_lora_weights_into_transformer(self.feedforward_model, args.lora_ckpt)

        if args.enable_xformers:
            if is_xformers_available():
                import xformers

                self.feedforward_model.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if self.gradient_checkpointing:
            self.feedforward_model.enable_gradient_checkpointing()

        self.feedforward_model.forward = types.MethodType(
            transformer_forward, self.feedforward_model
        )

        self.vae = AutoencoderKL.from_pretrained(
            args.model_id,
            subfolder="vae"
        ).float().to(accelerator.device)
        self.vae.requires_grad_(False)

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 16
        )

        if self.use_fp16:
            self.vae.to(torch.bfloat16)

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.model_id, subfolder="scheduler"
        )
        self.noise_scheduler_copy = copy.deepcopy(noise_scheduler)

        self.network_context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.use_fp16 else NoOpContext()

        if args.percep_weight > 0:
            self.precep_loss = pyiqa.create_metric(args.percep_loss, device=accelerator.device, as_loss=True)
            # Freeze metric parameters while retaining gradients to the predicted image.
            self.precep_loss.requires_grad_(False).eval()
            self.edge_detection_model = EdgeDetectionModel().to(accelerator.device)
            self.edge_detection_model.requires_grad_(False)
            if args.percep_weight > 0 and args.edge_weight == 0:
                del self.edge_detection_model


    def decode_image(self, latents):
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents).sample.float().clamp(-1, 1)
        return image

    def encode_image(self, image):
        latents = self.vae.encode(image.to(self.vae.dtype)).latent_dist.sample()
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return latents

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler_copy.sigmas.to(device=self.accelerator.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(self.accelerator.device)
        timesteps = timesteps.to(self.accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = rearrange(latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        return rearrange(
            latents,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=math.ceil(height / 16),
            w=math.ceil(width / 16),
            ph=2,
            pw=2,
        )

    def forward(self, lq, t5_embedding, clip_embedding,
        visual=False,
        real_train_dict=None,
    ):
        loss_dict = {}
        log_dict = {}
        bsz, c, height, width = lq.shape

        with self.network_context_manager:
            lq_latent = self.encode_image(lq * 2 - 1)

            latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    lq_latent.shape[0],
                    lq_latent.shape[2] // 2,
                    lq_latent.shape[3] // 2,
                    lq_latent.device,
                    lq_latent.dtype,
                )

            noise = real_train_dict['noise_latent']
            noise = self._unpack_latents(
                noise,
                height=lq_latent.shape[2] * self.vae_scale_factor,
                width=lq_latent.shape[3] * self.vae_scale_factor,
                vae_scale_factor=self.vae_scale_factor,
            )

            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling(
                weighting_scheme="logit_normal",
                batch_size=bsz,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=lq_latent.device)

            # Add noise according to flow matching.
            # zt = (1 - texp) * x + texp * z1
            sigmas = self.get_sigmas(timesteps, n_dim=lq_latent.ndim, dtype=lq_latent.dtype)

            T_lr = self.conditioning_timestep / 1000
            timesteps = timesteps * (1 - T_lr) + self.conditioning_timestep
            sigmas = sigmas * (1 - T_lr) + T_lr

            noisy_model_input = ((1.0 - sigmas) * lq_latent + (sigmas - T_lr) * noise) / (1 - T_lr)

            packed_noisy_model_input = self._pack_latents(
                noisy_model_input,
                batch_size=lq_latent.shape[0],
                num_channels_latents=lq_latent.shape[1],
                height=lq_latent.shape[2],
                width=lq_latent.shape[3],
            )

            guidance = torch.tensor([self.args.guidance], device=self.accelerator.device, dtype=torch.float32)
            guidance = guidance.expand(lq_latent.shape[0])

            model_pred = self.feedforward_model(
                hidden_states=packed_noisy_model_input,
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=clip_embedding,
                encoder_hidden_states=t5_embedding,
                txt_ids=torch.zeros(t5_embedding.shape[1], 3, device=self.accelerator.device),
                img_ids=latent_image_ids,
                return_dict=False,
                compute_adl=False,
            )[0]

            model_pred = self._unpack_latents(
                model_pred,
                height=lq_latent.shape[2] * self.vae_scale_factor,
                width=lq_latent.shape[3] * self.vae_scale_factor,
                vae_scale_factor=self.vae_scale_factor,
            )

            gt_latent = self._unpack_latents(
                real_train_dict['gt_latent'],
                height=lq_latent.shape[2] * self.vae_scale_factor,
                width=lq_latent.shape[3] * self.vae_scale_factor,
                vae_scale_factor=self.vae_scale_factor,
            )
            model_pred_teacher = noise - gt_latent

        target = noise - lq_latent

        # FTD (Eq. 17), using the stored teacher trajectory epsilon - z_0.
        v_pred = (model_pred_teacher - model_pred * T_lr)

        loss_fm = torch.mean(
            ((v_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss_fm = loss_fm.mean()

        loss_dict["loss_fm"] = loss_fm

        # Computing reconstruction loss
        with self.network_context_manager:
            packed_lq_latents = self._pack_latents(
                lq_latent,
                batch_size=lq_latent.shape[0],
                num_channels_latents=lq_latent.shape[1],
                height=lq_latent.shape[2],
                width=lq_latent.shape[3],
            )

            timestep_sr = torch.full((lq_latent.shape[0],), self.conditioning_timestep, dtype=lq_latent.dtype, device=lq_latent.device)

            scale_factor = 1.0
            if self.args.scale_variance:
                if self.args.dynamic_variance_scale_factor:
                    scale_factor = torch.sqrt(0.75 / torch.var(packed_lq_latents))
                else:
                    scale_factor = self.args.variance_scale_factor

            model_pred_sr, adl_loss = self.feedforward_model(
                hidden_states=packed_lq_latents * scale_factor,
                timestep=timestep_sr / 1000,
                guidance=guidance,
                pooled_projections=clip_embedding,
                encoder_hidden_states=t5_embedding,
                txt_ids=torch.zeros(t5_embedding.shape[1], 3, device=self.accelerator.device),
                img_ids=latent_image_ids,
                return_dict=False,
                compute_adl=self.args.adl_weight > 0,
            )
            model_pred_sr = self._unpack_latents(
                model_pred_sr,
                height=lq_latent.shape[2] * self.vae_scale_factor,
                width=lq_latent.shape[3] * self.vae_scale_factor,
                vae_scale_factor=self.vae_scale_factor,
            )

        pred_x0 = lq_latent.float() - T_lr * model_pred_sr.float()

        if self.args.adl_weight > 0:
            loss_dict["loss_adl"] = adl_loss

        with self.network_context_manager:
            pred_image = self.decode_image(pred_x0) * 0.5 + 0.5

        loss_spatial = 0
        if self.args.spatial_loss_type == "mse":
            mse_loss = F.mse_loss(pred_image, real_train_dict["gt_image"])
        else:
            mse_loss = F.l1_loss(pred_image, real_train_dict["gt_image"])
        loss_dict["loss_mse"] = mse_loss
        loss_spatial += mse_loss
        if self.args.percep_weight > 0:
            dists_loss = self.precep_loss(pred_image, real_train_dict["gt_image"])
            loss_dict["loss_percep"] = dists_loss
            loss_spatial += self.args.percep_weight * dists_loss
            if self.args.edge_weight > 0:
                edge_loss = self.precep_loss(
                    self.edge_detection_model(pred_image),
                    self.edge_detection_model(real_train_dict["gt_image"])
                )
                loss_dict["loss_edge"] = edge_loss
                loss_spatial +=self.args.edge_weight * edge_loss
            if self.args.tv_weight > 0:
                from utils.others import total_variation_loss
                tv_loss = self.precep_loss(
                    total_variation_loss(pred_image),
                    total_variation_loss(real_train_dict["gt_image"])
                )
                loss_dict["loss_tv"] = tv_loss
                loss_spatial +=self.args.tv_weight * tv_loss

        loss_dict["loss_spatial"] = loss_spatial

        if visual:
            with torch.no_grad():
                log_dict["pred_image"] = pred_image.detach()

        log_dict['denoising_timestep'] = self.conditioning_timestep

        return loss_dict, log_dict
