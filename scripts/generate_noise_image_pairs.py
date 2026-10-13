"""Generate FLUX noise/latent/image pairs directly into per-rank LMDBs.

Every prompt is used once, including the last incomplete batch. Seeds depend
only on the prompt index, and committed keys are skipped on restart. Point
training's --lmdb_path at --folder; no PT export or merge stage is needed.
The default prompt file is weights/fluxsr/prompts_10k.json.
"""

import argparse
import json
import pickle
import sys
import time
from datetime import timedelta
from pathlib import Path

import lmdb
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.prompt_dataset import PromptDataset


def sample_key(prompt_idx):
    return f"prompt_{prompt_idx:08d}".encode("ascii")


def completed_indices(folder):
    """Read keys only, without deserializing or copying image values."""
    completed = set()
    for shard in sorted(Path(folder).glob("rank*.lmdb")):
        if not (shard / "data.mdb").is_file():
            continue
        env = lmdb.open(str(shard), readonly=True, lock=False, readahead=False)
        try:
            with env.begin() as txn:
                for key in txn.cursor().iternext(keys=True, values=False):
                    if not key.startswith(b"prompt_"):
                        raise ValueError(f"Unexpected sample key in {shard}: {key!r}")
                    prompt_idx = int(key[len(b"prompt_"):])
                    if key != sample_key(prompt_idx) or prompt_idx in completed:
                        raise ValueError(f"Invalid or duplicate prompt key: {key!r}")
                    completed.add(prompt_idx)
        finally:
            env.close()
    return completed


def pending_indices(num_samples, rank, world_size, completed):
    # Redistribute unfinished prompts so resumed ranks have similar workloads.
    # Seeds remain tied to prompt_idx, independently of the writer shard.
    remaining = [i for i in range(num_samples) if i not in completed]
    return remaining[rank::world_size]


def make_noise(prompt_indices, seed, latent_channel, latent_resolution, device):
    # A separate generator per sample makes epsilon independent of batching,
    # worker count, and which earlier samples were already committed.
    noise = []
    for prompt_idx in prompt_indices:
        generator = torch.Generator(device=device).manual_seed(seed + prompt_idx)
        sample = torch.randn(
            (1, latent_channel, latent_resolution, latent_resolution),
            generator=generator, device=device, dtype=torch.float32,
        ).to(torch.bfloat16)
        noise.append(sample)
    return torch.cat(noise, dim=0)


def as_storage_array(tensor, expected_shape, name):
    array = tensor.detach().to(device="cpu", dtype=torch.float16).numpy()
    if array.shape != expected_shape or not np.isfinite(array).all():
        raise ValueError(f"Invalid {name}: shape {array.shape}, expected {expected_shape}, or nonfinite values")
    return array


@torch.inference_mode()
def generate_batch(pipeline, prompts, prompt_indices, args, device):
    batch_size = len(prompt_indices)
    resolution = args.latent_resolution * pipeline.vae_scale_factor
    noise = make_noise(
        prompt_indices, args.seed, args.latent_channel, args.latent_resolution, device,
    )
    initial_latents = pipeline._pack_latents(
        noise, batch_size, args.latent_channel, args.latent_resolution, args.latent_resolution,
    )
    # The archived epsilon must be the actual bf16 initial state used by FLUX.
    # Some schedulers mutate their sample: give the pipeline its own storage.
    output_latents = pipeline(
        prompt=prompts, latents=initial_latents.clone(), output_type="latent",
        height=resolution, width=resolution, guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps, num_images_per_prompt=1,
        max_sequence_length=args.max_sequence_length, return_dict=False,
    )[0]
    latent_shape = (batch_size, (args.latent_resolution // 2) ** 2, args.latent_channel * 4)
    input_arrays = as_storage_array(initial_latents, latent_shape, "input_latents")
    output_arrays = as_storage_array(output_latents, latent_shape, "output_latents")
    images = []
    # Decode in small batches: VAE activation memory scales with image area.
    for start in range(0, batch_size, args.decode_batch_size):
        stop = min(start + args.decode_batch_size, batch_size)
        latents = pipeline._unpack_latents(
            output_latents[start:stop], resolution, resolution, pipeline.vae_scale_factor,
        )
        latents = latents / pipeline.vae.config.scaling_factor + pipeline.vae.config.shift_factor
        decoded = pipeline.vae.decode(latents.to(pipeline.vae.dtype), return_dict=False)[0]
        pixels = pipeline.image_processor.postprocess(decoded, output_type="pt")
        images.extend(as_storage_array(
            pixels, (stop - start, 3, resolution, resolution), "images",
        ))
        del latents, decoded, pixels
    return [
        {
            "prompt_idx": prompt_idx,
            "seed": args.seed + prompt_idx,
            "input_latents": input_arrays[i],
            "output_latents": output_arrays[i],
            "images": images[i],
        }
        for i, prompt_idx in enumerate(prompt_indices)
    ]


def commit_batch(env, samples):
    """One durable transaction per inference batch; bounded CPU memory."""
    records = [(sample_key(s["prompt_idx"]), pickle.dumps(s, protocol=5)) for s in samples]
    while True:
        try:
            with env.begin(write=True) as txn:
                for key, value in records:
                    if not txn.put(key, value, overwrite=False):
                        raise ValueError(f"Refusing to overwrite existing pair {key!r}")
            return
        except lmdb.MapFullError:
            env.set_mapsize(env.info()["map_size"] * 2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True, help="Output directory containing rank*.lmdb")
    parser.add_argument("--prompt_path", default="weights/fluxsr/prompts_10k.json",
                        help="JSON list of strings or prompt dictionaries (default: weights/fluxsr/prompts_10k.json)")
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--decode_batch_size", type=int, default=1)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--latent_resolution", type=int, default=128)
    parser.add_argument("--latent_channel", type=int, default=16)
    parser.add_argument("--max_samples", type=int, help="Generate only the first N prompts, for a smoke run")
    parser.add_argument("--revision")
    args = parser.parse_args()
    for name in ["batch_size", "decode_batch_size", "num_inference_steps", "latent_resolution", "latent_channel"]:
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.latent_resolution % 2:
        parser.error("--latent_resolution must be even for FLUX latent packing")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max_samples must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    return args


def check_generation_settings(args, prompt_count, accelerator):
    # This small settings file prevents an accidental resume with a different
    # noise seed or teacher configuration. Batching/world size may change.
    settings = {
        "prompt_path": str(Path(args.prompt_path).resolve()),
        "prompt_count": prompt_count,
        "model_id": args.model_id,
        "revision": args.revision,
        "seed": args.seed,
        "latent_resolution": args.latent_resolution,
        "latent_channel": args.latent_channel,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "max_sequence_length": args.max_sequence_length,
        "storage_dtype": "float16",
    }
    path = Path(args.folder) / "generation_config.json"
    if accelerator.is_main_process:
        if path.exists():
            if json.loads(path.read_text()) != settings:
                raise ValueError(f"Generation settings changed; use a new --folder or restore {path}")
        else:
            path.write_text(json.dumps(settings, indent=2) + "\n")
    accelerator.wait_for_everyone()


def main():
    args = parse_args()
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs
    from diffusers import FluxPipeline

    # Independent sampling ranks can finish at different times; allow the
    # final barrier to wait for stragglers, including zero-pending resumes.
    accelerator = Accelerator(
        mixed_precision="no",
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=60))],
    )
    if accelerator.device.type != "cuda":
        raise RuntimeError("FLUX pair generation requires a CUDA GPU")
    folder = Path(args.folder)
    folder.mkdir(parents=True, exist_ok=True)
    prompts = PromptDataset(args.prompt_path)
    num_samples = len(prompts) if args.max_samples is None else min(args.max_samples, len(prompts))
    check_generation_settings(args, len(prompts), accelerator)
    completed = completed_indices(folder)
    if not completed.issubset(range(num_samples)):
        raise ValueError("Output contains prompt indices outside the requested sample range; use a new --folder")
    # All ranks must finish reading the same committed-key snapshot before
    # any rank can generate and commit newly redistributed work.
    accelerator.wait_for_everyone()
    rank = accelerator.process_index
    pending = pending_indices(num_samples, rank, accelerator.num_processes, completed)
    print(f"Rank {rank}: {len(pending)} remaining; {len(completed)}/{num_samples} already committed globally", flush=True)

    if pending:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        pipeline = FluxPipeline.from_pretrained(
            args.model_id, revision=args.revision, torch_dtype=torch.bfloat16,
        ).to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        # Normal FLUX.1-dev has 8x VAE compression and 16 latent channels.
        if pipeline.transformer.config.in_channels != args.latent_channel * 4:
            raise ValueError("--latent_channel does not match the FLUX transformer")
        resolution = args.latent_resolution * pipeline.vae_scale_factor
        bytes_per_sample = 2 * (3 * resolution ** 2 + 2 * args.latent_channel * args.latent_resolution ** 2)
        expected_rank_count = (num_samples + accelerator.num_processes - 1) // accelerator.num_processes
        map_size = max(64 * 1024 ** 2, int(bytes_per_sample * expected_rank_count * 1.15))
        shard_path = folder / f"rank{rank:05d}.lmdb"
        data_path = shard_path / "data.mdb"
        if data_path.exists():
            map_size = max(map_size, int(data_path.stat().st_size * 1.15))
        env = lmdb.open(str(shard_path), map_size=map_size, sync=True, metasync=True)
        torch.cuda.reset_peak_memory_stats(accelerator.device)
        generation_start = time.perf_counter()
        generated = 0
        try:
            progress = tqdm(total=len(pending), desc=f"rank {rank}", disable=not accelerator.is_local_main_process)
            for batch_idx, start in enumerate(range(0, len(pending), args.batch_size), start=1):
                batch_start = time.perf_counter()
                indices = pending[start:start + args.batch_size]
                samples = generate_batch(pipeline, [prompts[i] for i in indices], indices, args, accelerator.device)
                commit_batch(env, samples)
                generated += len(samples)
                progress.update(len(samples))
                del samples
                if batch_idx == 1 or batch_idx % 10 == 0:
                    elapsed = time.perf_counter() - generation_start
                    peak_gib = torch.cuda.max_memory_allocated(accelerator.device) / 1024 ** 3
                    print(
                        f"Rank {rank} batch {batch_idx}: generated={generated}/{len(pending)} "
                        f"batch_seconds={time.perf_counter() - batch_start:.2f} "
                        f"elapsed_seconds={elapsed:.2f} samples_per_second={generated / elapsed:.4f} "
                        f"cuda_peak_GiB={peak_gib:.2f}", flush=True,
                    )
            progress.close()
        finally:
            env.close()
        elapsed = time.perf_counter() - generation_start
        print(
            f"Rank {rank} summary: generated={generated} elapsed_seconds={elapsed:.2f} "
            f"samples_per_second={generated / elapsed:.4f} "
            f"cuda_peak_GiB={torch.cuda.max_memory_allocated(accelerator.device) / 1024 ** 3:.2f}",
            flush=True,
        )
        del pipeline
    else:
        print(f"Rank {rank} summary: generated=0 (all assigned keys already committed)", flush=True)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        completed = completed_indices(folder)
        if completed != set(range(num_samples)):
            raise RuntimeError(f"Expected {num_samples} unique prompt keys, found {len(completed)}")
        summary = {
            "samples": num_samples, "shards": len(list(folder.glob("rank*.lmdb"))),
            "latent_shape": [(args.latent_resolution // 2) ** 2, args.latent_channel * 4],
            "image_shape": [3, args.latent_resolution * 8, args.latent_resolution * 8],
            "storage_dtype": "float16", "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
        }
        (folder / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Completed: {json.dumps(summary)}", flush=True)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
