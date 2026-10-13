---
language:
  - en
license: other
license_name: flux-1-dev-non-commercial-license
license_link: https://huggingface.co/mon3tr/FluxSR-private/blob/main/FLUX.1-dev-LICENSE.md
base_model: black-forest-labs/FLUX.1-dev
base_model_relation: adapter
library_name: diffusers
pipeline_tag: image-to-image
tags:
  - super-resolution
  - flux
  - lora
  - image-restoration
  - icml-2025
---

# FluxSR

FluxSR is a LoRA adapter for one-step real-world image super-resolution,
based on [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev).
Training, inference, and evaluation code is publicly available in the
[FluxSR project](https://github.com/JianzeLi-114/FluxSR).
The adapter is provided as `pytorch_lora_weights.safetensors`.

## Files and use

- `pytorch_lora_weights.safetensors`: transformer LoRA adapter.
- `training_config.json`: training and inference settings.
- `t5_embedding.pth`, `clip_embedding.pth`: the fixed text conditions.
- `prompts_10k.json`: the 10,000 prompts used to generate training pairs.
- `FLUX.1-dev-LICENSE.md`: model license supplied with the base model.
- `NOTICE`: BFL attribution and FluxSR modification notices.

Use the FluxSR inference entry point described in the project's README,
with this directory as the adapter path. The FLUX.1-dev base model and
the included fixed T5/CLIP text embeddings are also needed.
Download the base model from BFL after accepting its access conditions.
The model package provides the LoRA weights, configuration, fixed text
embeddings, and training prompts. Model binaries and JSON files are hosted
on Hugging Face and are not tracked in the GitHub code repository.

From the FluxSR code directory:

```bash
python inference.py \
  --input /path/to/input.png --output results/example \
  --model /path/to/FLUX.1-dev --lora weights/fluxsr --seed 114
```

The input can also be a directory. Inference performs ×4 super-resolution
and reads timestep, guidance, LoRA rank, and alpha from
`training_config.json`. The fixed embeddings are loaded from the adapter
directory. CPU offload (`--offload sequential`) and VAE tiling
(`--vae-tiling`) are optional. The evaluation below uses full-image inference.

This model repository is **public**. Download the complete package:

```bash
hf download mon3tr/FluxSR-private --local-dir weights/fluxsr
```

The package contains the FluxSR adapter and supporting files. Download
the FLUX.1-dev base model separately from Black Forest Labs.

## Training configuration

| Setting | Value |
|---|---|
| Base and teacher | FLUX.1-dev |
| Generated pairs | 10,000 prompts, 1024 × 1024 |
| Training steps | 20,000 |
| Teacher sampling | 50 steps, guidance 3.5, base seed 10 |
| Training batch | 1 per GPU, 8 GPUs; global batch 8 |
| LoRA rank / alpha | 64 / 64 |
| Learning rate / warmup | 1e-5 / 500 steps |
| Student conditioning timestep / guidance | 350 / 4.0 |
| Losses | Spatial MSE FTD, LPIPS 1.0, TV-LPIPS 1.0, ADL 0.1 |
| Training precision | BF16 frozen weights; FP32 trainable adapters |
| Training seed | 114 |

ADL averages the cosine similarity between each image token and the mean
image-token representation across all 57 transformer blocks.

## RealSR ×4 evaluation

Evaluation of the supplied weights:

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | MUSIQ ↑ | MANIQA ↑ | TOPIQ-NR ↑ | Q-Align ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FluxSR | 25.15367 | 0.73374 | 0.2964075 | 0.1870468 | 68.1300 | 0.5086289 | 0.6266896 | 4.382539 |

The evaluation uses all 100 Canon/Nikon test pairs from RealSR (V3),
scale 4, full-image inference with BF16 frozen base weights and FP32 LoRA
parameters, timestep 350, guidance 4, and seed 114
with the sorted image index added per image. No color correction or
tiled inference was used. PSNR and SSIM use the Y channel with border
crop 0; the other metrics use the documented PyIQA defaults. Refer to
the project README for the exact evaluation settings and separate
Q-Align environment.

## Intended use and limitations

Intended for non-commercial research on real-world image super-resolution
under the accompanying model license. As a generative restoration
model, FluxSR may synthesize details that are absent from the input;
restored content should not be treated as factual evidence. Large
images require more GPU memory. Changing precision, tiling, image
preprocessing, or random seeds may change outputs and measured scores.

## License and attribution

The adapter is a modification of FLUX.1-dev. The adapter and bundled
fixed T5/CLIP text-embedding tensors are subject to the
[FLUX.1 \[dev\] Non-Commercial License](FLUX.1-dev-LICENSE.md).
See [NOTICE](NOTICE) for the required BFL attribution and modification
notice. Rights to the base model and derivatives are granted directly
by BFL under that license. The Apache-2.0 license of the project code
does not grant additional rights to these model assets. This adapter is not
an official BFL product and is not endorsed by BFL.

## Citation

```bibtex
@inproceedings{li2025one,
  title={One Diffusion Step to Real-World Super-Resolution via Flow Trajectory Distillation},
  author={Li, Jianze and Cao, Jiezhang and Guo, Yong and Li, Wenbo and Zhang, Yulun},
  booktitle={ICML},
  year={2025}
}
```
