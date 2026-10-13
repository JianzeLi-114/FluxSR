<div align="center">

<h1>FluxSR</h1>
<h3>One Diffusion Step to Real-World Super-Resolution<br>via Flow Trajectory Distillation</h3>

<p><strong>ICML 2025</strong></p>

<p>
  Jianze Li,
  <a href="https://www.jiezhangcao.com/">Jiezhang Cao</a>,
  <a href="https://www.guoyongcs.com/">Yong Guo</a>,
  <a href="https://fenglinglwb.github.io/">Wenbo Li</a>, and
  <a href="http://yulunzhang.com/">Yulun Zhang<sup>*</sup></a>
</p>

<p>
  <a href="https://arxiv.org/abs/2502.01993"><img src="https://img.shields.io/badge/arXiv-2502.01993-b31b1b.svg" alt="arXiv paper"></a>
  <a href="https://huggingface.co/mon3tr/FluxSR-private"><img src="https://img.shields.io/badge/Models-Hugging%20Face-FFD21E?logo=huggingface&amp;logoColor=FFD21E" alt="Public Hugging Face models"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code%20License-Apache%202.0-blue.svg" alt="Code license: Apache 2.0"></a>
</p>

<p>
  <a href="https://github.com/JianzeLi-114/FluxSR"><img src="https://visitor-badge.laobi.icu/badge?page_id=JianzeLi-114/FluxSR" alt="Project page visits"></a>
  <a href="https://github.com/JianzeLi-114/FluxSR/stargazers"><img src="https://img.shields.io/github/stars/JianzeLi-114/FluxSR?style=social&amp;label=Public%20repo%20stars" alt="Public FluxSR repository stars"></a>
  <a href="https://github.com/JianzeLi-114/FluxSR/forks"><img src="https://img.shields.io/github/forks/JianzeLi-114/FluxSR?style=social&amp;label=Public%20repo%20forks" alt="Public FluxSR repository forks"></a>
  <a href="https://huggingface.co/mon3tr/FluxSR-private/tree/main"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fmon3tr%2FFluxSR-private&amp;query=%24.downloads&amp;label=HF%20downloads%2Fmonth&amp;color=FFD21E&amp;logo=huggingface" alt="Hugging Face model downloads per month"></a>
</p>

<p>
  <strong>English</strong> &nbsp;|&nbsp; <a href="README_ZH.md">简体中文</a>
</p>

<p>
  <a href="https://proceedings.mlr.press/v267/li25a.html">Paper</a> &nbsp;·&nbsp;
  <a href="https://arxiv.org/abs/2502.01993">arXiv</a> &nbsp;·&nbsp;
  <a href="#models">Pretrained Models</a> &nbsp;·&nbsp;
  <a href="#training">Training</a> &nbsp;·&nbsp;
  <a href="#testing">Testing</a> &nbsp;·&nbsp;
  <a href="#results">Results</a>
</p>

</div>

#### 🔥 News

- **2026-10-13:** The training, inference, and evaluation code is now available, together with the [Hugging Face model package](https://huggingface.co/mon3tr/FluxSR-private) and a [Chinese README](README_ZH.md).
- **2025-02-03:** This repo is released.

---

> **Abstract:** Diffusion models (DMs) have significantly advanced the development of real-world image super-resolution (Real-ISR), but the computational cost of multi-step diffusion models limits their application. One-step diffusion models generate high-quality images in a one sampling step, greatly reducing computational overhead and inference latency. However, most existing one-step diffusion methods are constrained by the performance of the teacher model, where poor teacher performance results in image artifacts. To address this limitation, we propose FluxSR, a novel one-step diffusion Real-ISR technique based on flow matching models. We use the state-of-the-art diffusion model FLUX.1-dev as both the teacher model and the base model. First, we introduce Flow Trajectory Distillation (FTD) to distill a multi-step flow matching model into a one-step Real-ISR. Second, to improve image realism and address high-frequency artifact issues in generated images, we propose TV-LPIPS as a perceptual loss and introduce Attention Diversification Loss (ADL) as a regularization term to reduce token similarity in transformer, thereby eliminating high-frequency artifacts. Comprehensive experiments demonstrate that our method outperforms existing one-step diffusion-based Real-ISR methods.

<p align="center">
  <img src="figs/teaser.png" width="900" alt="Visual comparisons of FluxSR with real-world image super-resolution methods">
</p>

---

### Pipeline

<p align="center">
  <img src="figs/pipeline.png" width="900" alt="FluxSR pipeline: Flow Trajectory Distillation, TV-LPIPS, and Attention Diversification Loss">
</p>

---

## 🔗 Contents

- [Dependencies](#dependencies) · [Models](#models) · [Datasets](#datasets)
- [Training](#training) · [Testing](#testing) · [Evaluation](#evaluation)
- [Results](#results) · [Publishing](#publishing) · [Citation](#citation) · [Acknowledgements](#acknowledgements) · [License](#license)


<a id="dependencies"></a>

## ⚙️ Dependencies

- Python 3.11
- PyTorch 2.5.1 / Torchvision 0.20.1 / CUDA 12.4
- Diffusers, Transformers, PEFT, Accelerate, and PyIQA

Run commands from the repository root.

```bash
git clone https://github.com/JianzeLi-114/FluxSR.git
cd FluxSR
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Requirements pin Diffusers 0.37.1, Transformers 5.5.0, PEFT 0.18.1,
Accelerate 1.13.0, and PyIQA 0.1.15.post2. Q-Align uses a separate
environment described under [Evaluation](#evaluation).

<a id="models"></a>

## 📦 Models

| Model | Description | Download | Local path |
|:---|:---|:---|:---|
| FLUX.1-dev | Base model and teacher | [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-dev) | `pretrained/FLUX.1-dev/` |
| **FluxSR** | LoRA weights, configuration, fixed text embeddings, and training prompts | [Hugging Face](https://huggingface.co/mon3tr/FluxSR-private) | `weights/fluxsr/` |

Download the **base model** from
[black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
after accepting its access conditions. Base weights are separately
licensed and are not bundled here.

```bash
hf auth login
hf download black-forest-labs/FLUX.1-dev --local-dir pretrained/FLUX.1-dev
export MODEL_ID="$PWD/pretrained/FLUX.1-dev"
```

A previously downloaded directory or the authenticated Hugging Face model
ID can also be used wherever `MODEL_ID` is shown.

The [FluxSR model package](https://huggingface.co/mon3tr/FluxSR-private) is public.
Download the complete package:

```bash
export HF_MODEL_REPO="mon3tr/FluxSR-private"
hf download "$HF_MODEL_REPO" --local-dir weights/fluxsr
```

See the [model card](weights/fluxsr/README.md) for configuration and evaluation
results. The adapter directory contains the LoRA weights, `training_config.json`,
fixed text embeddings, `prompts_10k.json`, and model license. These model
files and JSON files are hosted on Hugging Face rather than tracked in GitHub.

```text
weights/fluxsr/
├── pytorch_lora_weights.safetensors
├── training_config.json
├── t5_embedding.pth
├── clip_embedding.pth
├── prompts_10k.json
├── README.md
├── FLUX.1-dev-LICENSE.md
└── NOTICE
```

<a id="datasets"></a>

## 📁 Datasets

| Split | Dataset | Preparation | Local path |
|:---|:---|:---|:---|
| Training | FLUX-generated noise–image pairs | [Generate pairs](#training-data) | `data/flux_pairs_10k/` |
| Testing | Your low-resolution images, with optional ground truth | [Directory layout](#test-data) | `data/example/` |

<a id="training-data"></a>

### Training data

The model package's `weights/fluxsr/prompts_10k.json` contains 10,000 prompts.
Generate the training pairs with the FLUX.1-dev teacher:

```bash
MODEL_ID="$MODEL_ID" NPROC_PER_NODE=8 bash scripts/generate_data.sh
```

This creates sharded LMDB data under `data/flux_pairs_10k`. Each record
contains paired initial noise, the teacher output latent, the decoded
1024 × 1024 image, prompt index, and seed. The teacher uses 50 steps,
guidance 3.5, and seed `10 + prompt_index`. Generation uses BF16 model
weights and stores latent/image arrays in FP16.

<details>
<summary><strong>Generation options</strong></summary>

Defaults are 4 prompts per GPU and VAE decode batches of 1. Generation
distributes remaining prompts across workers and resumes by skipping
existing prompt keys. Paths and batch size can be changed:

```bash
MODEL_ID="$MODEL_ID" \
PROMPT_PATH=weights/fluxsr/prompts_10k.json LMDB_PATH=data/flux_pairs_10k \
NPROC_PER_NODE=8 GEN_BATCH_SIZE=4 \
bash scripts/generate_data.sh
```

For a small first run, add `MAX_SAMPLES=8`. Rerun without it to finish
all 10,000 prompts in the same directory. Use a new output directory
when changing the prompts, teacher, or generation settings.

</details>

<a id="test-data"></a>

### Test data

Place your low-resolution images under `input/`. For full-reference
evaluation, provide the corresponding high-resolution images under `gt/`:

```text
data/example/
├── input/
│   ├── image001.png
│   └── image002.png
└── gt/
    ├── image001.png
    └── image002.png
```

Ground-truth images should match the prediction dimensions. If ground
truth is unavailable, use the no-reference evaluation command below.

<a id="training"></a>

## 🔧 Training

Train FluxSR from the base model:

```bash
MODEL_ID="$MODEL_ID" LMDB_PATH=data/flux_pairs_10k \
OUTPUT_DIR=exp/fluxsr NPROC_PER_NODE=8 \
bash train.sh
```

> 💡 Training uses eight A800 80 GB GPUs, with one image per GPU.
> Checkpoints are saved every 1,000 steps; every 5,000-step checkpoint is kept permanently.

Allow about 80 GB of disk space for the generated LMDB pairs.

<details>
<summary><strong>Training configuration and checkpoint resume</strong></summary>

| Setting | Default |
|---|---|
| Training pairs | All 10,000 generated pairs |
| Training steps | 20,000 |
| Batch size | 1 per GPU; global batch 8 |
| LoRA rank / alpha | 64 / 64 |
| Learning rate / warmup | 1e-5 / 500 steps |
| Conditioning timestep / guidance | 350 / 4.0 |
| Losses | Spatial MSE FTD, LPIPS 1.0, TV-LPIPS 1.0, ADL 0.1 |
| Precision | BF16 frozen weights; FP32 trainable adapters |
| Gradient checkpointing | Enabled |
| Training seed | 114 |
| Degradation configuration | `dataset/realesrgan_paper.yaml` |

ADL averages the cosine similarity between each image token and the mean
image-token representation across all 57 transformer blocks.

Set `TRAIN_ITERS` to change the training length. Changing `NPROC_PER_NODE`
also changes the global batch size. Resume a full checkpoint,
including optimizer, scheduler, and RNG state, with:

```bash
MODEL_ID="$MODEL_ID" LMDB_PATH=data/flux_pairs_10k \
OUTPUT_DIR=exp/fluxsr \
RESUME_FROM=/path/to/full/checkpoint_010000 \
bash train.sh
```

Checkpoints are stored under
`exp/fluxsr/<timestamp>_seed114/checkpoint_<step>/`; use the full path
to the selected checkpoint. Resume restores the saved optimizer step and
training state. The `weights/fluxsr` package contains the LoRA weights and
configuration for inference.

</details>

<a id="testing"></a>

## 🔨 Testing

Run ×4 super-resolution on a single image or a directory:

```bash
python inference.py \
  --input /path/to/input.png \
  --output results/example \
  --model "$MODEL_ID" \
  --lora weights/fluxsr \
  --seed 114
```

> 💡 Use `--offload sequential` to reduce GPU memory usage and `--vae-tiling`
> to decode large images in tiles. Run `python inference.py --help` for all options.

<details>
<summary><strong>Inference settings</strong></summary>

Pass a directory to `--input` to process its images. Output is always a
directory. Inference performs ×4 super-resolution; timestep 350,
guidance 4, and LoRA rank/alpha 64/64 are read from the supplied
`training_config.json`.

The evaluation uses full-image inference, BF16 frozen base weights with
FP32 LoRA parameters, no color correction, and no VAE tiling. Sampling uses
`seed + image_index` over the sorted input list; keep the complete input list to reproduce the
benchmark. For a single image taken from that list, set `--image-index`
to its original zero-based index to use the same seed.

The evaluation commands below use full-image inference. Fixed text
embeddings are loaded from `weights/fluxsr/`. Inputs whose shorter side is below
128 pixels are enlarged before inference and resized back to exactly
4 times the input dimensions afterward. Use `--t5-embedding` and
`--clip-embedding` to supply other file locations.

</details>

<a id="evaluation"></a>

## 📏 Evaluation

Run inference on your input images, then evaluate predictions with or
without ground truth.

```bash
python inference.py \
  --input data/example/input \
  --output results/example/pred \
  --model "$MODEL_ID" \
  --lora weights/fluxsr \
  --seed 114

python scripts/evaluate.py \
  --pred-dir results/example/pred \
  --gt-dir data/example/gt \
  --output-dir results/example/metrics \
  --color-space y \
  --fail-on-metric-error
```

With `--gt-dir`, the default metrics are PSNR, SSIM, LPIPS, DISTS, MUSIQ,
MANIQA, and TOPIQ-NR. Without ground truth, the default metrics are MUSIQ,
MANIQA, and TOPIQ-NR:

```bash
python scripts/evaluate.py \
  --pred-dir results/example/pred \
  --output-dir results/example/metrics_nr \
  --fail-on-metric-error
```

The evaluator writes `per_image.csv` and `summary.json`. Predictions and
references must have matching dimensions; evaluation does not resize or
align them.

<details>
<summary><strong>Pairing and metric options</strong></summary>

By default, images are paired by relative path without the extension.
For example, `scene/image001.png` matches `scene/image001.jpg` under the
prediction and ground-truth directories.

- `--match-mode relative`: match relative paths without extensions (default).
- `--match-mode stem`: match filenames without directories or extensions.
- `--pred-suffix` and `--gt-suffix`: remove a filename suffix before matching.

For filenames such as `image001_SR.png` and `image001_HR.png`:

```bash
python scripts/evaluate.py \
  --pred-dir /path/to/predictions \
  --gt-dir /path/to/ground_truth \
  --output-dir results/metrics \
  --match-mode stem --pred-suffix _SR --gt-suffix _HR \
  --color-space y \
  --fail-on-metric-error
```

PSNR and SSIM default to YCbCr Y, crop border 0, and no SSIM downsampling.
Use `--color-space rgb` to calculate PSNR and SSIM in RGB. LPIPS uses
AlexNet/v0.1 and TOPIQ uses the no-reference variant; other metric options
follow PyIQA defaults. Select individual metrics with `--metrics`.

</details>

<details>
<summary><strong>Q-Align evaluation</strong></summary>

Q-Align uses a separate environment with PyTorch 2.6.0, Torchvision 0.21.0,
Transformers 4.46.2, and PyIQA 0.1.15.post2:

```bash
python3.11 -m venv .venv-qalign
.venv-qalign/bin/pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.venv-qalign/bin/pip install transformers==4.46.2 pyiqa==0.1.15.post2 icecream==2.1.10 numpy==1.26.4 opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86 "Pillow>=10.0,<13"
.venv-qalign/bin/python scripts/evaluate.py \
  --pred-dir results/example/pred \
  --output-dir results/example/metrics_qalign \
  --metrics qalign --fail-on-metric-error
```

PyIQA downloads metric weights on first use.

</details>

<a id="results"></a>

## 🔎 Results

<details>
<summary><strong>Quantitative results from the paper</strong></summary>

Table 1 of the main paper:

<p align="center">
  <img src="figs/table1.png" width="900" alt="Table 1: quantitative comparisons from the FluxSR paper">
</p>

Table 2 of the main paper (RealSet65):

<p align="center">
  <img src="figs/table2.png" width="450" alt="Table 2: quantitative comparisons on RealSet65">
</p>

</details>

<details>
<summary><strong>Qualitative results from the paper</strong></summary>

Figure 5 of the main paper:

<p align="center">
  <img src="figs/visual.png" width="900" alt="Figure 5: qualitative comparisons from the FluxSR paper">
</p>

</details>

<a id="publishing"></a>

## 🚀 Publishing

<details>
<summary><strong>Publish the model package and source code</strong></summary>

For maintainers with write access, update the existing public model
repository with the complete model package:

```bash
hf auth login
export HF_MODEL_REPO="mon3tr/FluxSR-private"
hf upload "$HF_MODEL_REPO" weights/fluxsr . --repo-type model
```

The destination model repository is public. See the
[official Hugging Face upload guide](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-upload).

The public GitHub source repository is
[JianzeLi-114/FluxSR](https://github.com/JianzeLi-114/FluxSR).
It provides the training, inference, and evaluation code. Model binaries
and JSON files are hosted separately in the public Hugging Face package
and are excluded from Git.

</details>

<a id="citation"></a>

## 📎 Citation

```bibtex
@inproceedings{li2025one,
  title={One Diffusion Step to Real-World Super-Resolution via Flow Trajectory Distillation},
  author={Li, Jianze and Cao, Jiezhang and Guo, Yong and Li, Wenbo and Zhang, Yulun},
  booktitle={ICML},
  year={2025}
}
```

<a id="acknowledgements"></a>

## 💡 Acknowledgements

This project builds on [FLUX](https://github.com/black-forest-labs/flux),
[Diffusers](https://github.com/huggingface/diffusers),
[PEFT](https://github.com/huggingface/peft), and
[BasicSR](https://github.com/XPixelGroup/BasicSR).
Evaluation uses [PyIQA](https://github.com/chaofengc/IQA-PyTorch).

<a id="license"></a>

## 📄 License

Project code is licensed under [Apache-2.0](LICENSE). The FLUX.1-dev base
model, FluxSR LoRA weights, and bundled fixed T5/CLIP text embeddings are
subject to the separate
[FLUX.1 \[dev\] Non-Commercial License](licenses/FLUX.1-dev-LICENSE.md);
the code license does not relicense these model assets. See [NOTICE](NOTICE)
for BFL attribution and the modification statement, and
[license notes](licenses/README.md) for the supplied license and current
upstream links.
