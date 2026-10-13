#!/usr/bin/env python3
"""FluxSR x4 inference for a single image or a directory.

Use --offload sequential to reduce GPU memory usage.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF
from diffusers import AutoencoderKL, FluxPipeline, FluxTransformer2DModel
from peft import LoraConfig
from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file

TARGET_MODULES = [
    "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2",
]


def load_fp32_adapter(transformer, weight_path, config):
    """Load LoRA weights into FP32 adapter parameters."""
    transformer.requires_grad_(False)
    transformer.add_adapter(LoraConfig(
        r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        lora_dropout=config.get("lora_dropout", 0.0), target_modules=TARGET_MODULES,
    ))
    for name, parameter in transformer.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.float()
    state = load_file(str(weight_path), device="cpu")
    if not state or any(not key.startswith("transformer.") for key in state):
        raise ValueError("Expected transformer-only diffusers LoRA weights")
    state = {key.removeprefix("transformer."): value for key, value in state.items()}
    incompatible = set_peft_model_state_dict(transformer, state, adapter_name="default")
    missing = [key for key in incompatible.missing_keys if "lora_" in key]
    if missing or incompatible.unexpected_keys:
        raise ValueError(f"LoRA mismatch: missing={missing}, unexpected={incompatible.unexpected_keys}")
    transformer.requires_grad_(False).eval()


def prepare_image(path):
    with Image.open(path) as source:
        image = source.convert("RGB")
    original_size = image.size
    resized_small = min(image.size) < 128
    if resized_small:
        factor = 128 / min(image.size)
        image = image.resize((int(factor * image.width), int(factor * image.height)),
                             Image.Resampling.BICUBIC)
    image = image.resize((image.width * 4, image.height * 4), Image.Resampling.BICUBIC)
    crop_size = image.size
    image = ImageOps.expand(image, (0, 0, (-image.width) % 16, (-image.height) % 16),
                            fill=(0, 0, 0))
    return image, original_size, crop_size, resized_small


def finish_image(tensor, original_size, crop_size, resized_small):
    image = TF.to_pil_image(tensor[0].float().cpu())
    image = image.crop((0, 0, *crop_size))
    if resized_small:
        image = image.resize((original_size[0] * 4, original_size[1] * 4),
                             Image.Resampling.LANCZOS)
    return image


def seed_image(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def embedding_path(explicit_path, checkpoint_dir, filename):
    """Resolve an explicit embedding path or the file in the LoRA directory."""
    path = Path(explicit_path) if explicit_path is not None else Path(checkpoint_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Text embedding file not found: {path}. Place {filename} in the LoRA directory "
            "or specify --t5-embedding and --clip-embedding explicitly."
        )
    return path


class FluxSR:
    def __init__(self, args):
        self.device = torch.device(args.device)
        if self.device.type not in {"cuda", "cpu"}:
            raise ValueError("Supported devices are CUDA and CPU")
        if args.offload != "none" and self.device.type != "cuda":
            raise ValueError("Sequential CPU offload requires a CUDA execution device")
        checkpoint = Path(args.lora)
        checkpoint_dir = checkpoint if checkpoint.is_dir() else checkpoint.parent
        weight_path = (checkpoint / "pytorch_lora_weights.safetensors"
                       if checkpoint.is_dir() else checkpoint)
        config_path = checkpoint_dir / "training_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Checkpoint configuration not found: {config_path}")
        self.config = json.loads(config_path.read_text())
        if self.config.get("train_vae", False):
            raise ValueError("This entry point expects a frozen base-model VAE")
        t5_path = embedding_path(args.t5_embedding, checkpoint_dir, "t5_embedding.pth")
        clip_path = embedding_path(args.clip_embedding, checkpoint_dir, "clip_embedding.pth")
        print(f"Loading {args.model}; t={self.config['conditioning_timestep']}, "
              f"guidance={self.config['guidance']}, BF16 base / FP32 LoRA", flush=True)
        # Initialize on CPU for sequential offload; device-only .to() preserves
        # FP32 adapter parameters.
        self.transformer = FluxTransformer2DModel.from_pretrained(
            args.model, subfolder="transformer", torch_dtype=torch.bfloat16,
        )
        load_fp32_adapter(self.transformer, weight_path, self.config)
        self.vae = AutoencoderKL.from_pretrained(
            args.model, subfolder="vae", torch_dtype=torch.bfloat16,
        ).eval().requires_grad_(False)
        self.vae.disable_tiling()
        self.vae.disable_slicing()
        if args.vae_tiling:
            self.vae.enable_tiling()
        self.t5 = torch.load(t5_path, map_location="cpu", weights_only=True).to(self.device)
        self.clip = torch.load(clip_path, map_location="cpu", weights_only=True).to(self.device)
        self.scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        if args.offload == "sequential":
            from accelerate import cpu_offload
            cpu_offload(self.transformer, execution_device=self.device)
            cpu_offload(self.vae, execution_device=self.device)
        else:
            self.transformer.to(self.device)
            self.vae.to(self.device)

    @torch.inference_mode()
    def predict(self, image, seed):
        seed_image(seed)
        lq = TF.to_tensor(image).unsqueeze(0).to(self.device)
        latent = self.vae.encode((lq * 2 - 1).to(torch.bfloat16)).latent_dist.sample()
        latent = (latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        del lq
        batch, channels, height, width = latent.shape
        packed = FluxPipeline._pack_latents(latent, batch, channels, height, width)
        image_ids = FluxPipeline._prepare_latent_image_ids(
            batch, height // 2, width // 2, self.device, latent.dtype)
        timestep = torch.full((batch,), self.config["conditioning_timestep"],
                              dtype=latent.dtype, device=self.device)
        guidance = torch.full((batch,), self.config["guidance"],
                              dtype=torch.float32, device=self.device)
        num_train_timesteps = self.config.get("num_train_timesteps", 1000)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            prediction = self.transformer(
                hidden_states=packed,
                timestep=timestep / num_train_timesteps,
                guidance=guidance,
                pooled_projections=self.clip,
                encoder_hidden_states=self.t5,
                txt_ids=torch.zeros(self.t5.shape[1], 3, device=self.device),
                img_ids=image_ids,
                return_dict=False,
            )[0]
        prediction = FluxPipeline._unpack_latents(
            prediction, height=height * self.scale_factor,
            width=width * self.scale_factor, vae_scale_factor=self.scale_factor)
        ratio = self.config["conditioning_timestep"] / num_train_timesteps
        # Compute the residual and VAE inverse normalization in FP32.
        pred_x0 = latent.float() - ratio * prediction.float()
        del latent, packed, prediction
        decode_latent = pred_x0 / self.vae.config.scaling_factor + self.vae.config.shift_factor
        del pred_x0
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            output = self.vae.decode(decode_latent).sample.float().clamp(-1, 1) * 0.5 + 0.5
        if not torch.isfinite(output).all():
            raise ValueError("Non-finite model output")
        return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--input_image", "-i", required=True, type=Path,
                        help="An image or a directory of PNG/JPEG images")
    parser.add_argument("--output", "--output_dir", "-o", required=True, type=Path,
                        help="Output directory; predictions are saved as PNG")
    parser.add_argument("--model", "--pretrained_model_name_or_path",
                        default="black-forest-labs/FLUX.1-dev",
                        help="FLUX.1-dev repository ID or local model directory")
    parser.add_argument("--lora", "--ckpt_path", default="weights/fluxsr",
                        help="Checkpoint directory, or safetensors file with adjacent training_config.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offload", choices=["none", "sequential"], default="none",
                        help="Sequential offload keeps weights on CPU between layer calls")
    parser.add_argument("--vae-tiling", "--vae_tiling", action="store_true",
                        help="Enable VAE tiling to reduce GPU memory usage; disabled by default")
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--image-index", type=int, default=0,
                        help="Single-image global index; directory images use their sorted index")
    parser.add_argument("--t5-embedding", "--t5_ckpt_path", type=Path)
    parser.add_argument("--clip-embedding", "--clip_ckpt_path", type=Path)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.image_index < 0 or (args.input.is_dir() and args.image_index != 0):
        parser.error("--image-index must be nonnegative and is only for single-image input")
    return args


def main():
    args = parse_args()
    if args.input.is_dir():
        names = sorted(path for path in args.input.iterdir()
                       if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"})
        images = list(enumerate(names))
    else:
        images = [(args.image_index, args.input)]
    if not images or any(not path.is_file() for _, path in images):
        raise ValueError(f"No input images found: {args.input}")
    stems = [path.stem for _, path in images]
    if len(stems) != len(set(stems)):
        raise ValueError("Input filenames must have unique stems for PNG output")
    args.output.mkdir(parents=True, exist_ok=True)
    model = FluxSR(args)
    for index, path in images:
        destination = args.output / (path.stem + ".png")
        if args.skip_existing and destination.is_file():
            print(f"Skipping {destination}", flush=True)
            continue
        prepared, original_size, crop_size, resized_small = prepare_image(path)
        started = time.monotonic()
        output = model.predict(prepared, args.seed + index)
        image = finish_image(output, original_size, crop_size, resized_small)
        del output
        image.save(destination)
        print(f"Saved {destination} {image.size}, seed={args.seed + index}, "
              f"{time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
