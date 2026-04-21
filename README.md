# OmniHuman: A Large-scale Dataset and Benchmark for Human-Centric Video Generation

# OmniHuman

[**Paper**](https://arxiv.org/abs/2604.18326) | [**arXiv PDF**](https://arxiv.org/pdf/2604.18326)

Official repository for **OmniHuman: A Large-scale Dataset and Benchmark for Human-Centric Video Generation**.

## Abstract

Recent advancements in audio-video joint generation models have demonstrated impressive capabilities in content creation. However, generating high-fidelity human-centric videos in complex, real-world physical scenes remains a significant challenge. We identify that the root cause lies in the structural deficiencies of existing datasets across three dimensions: limited global scene and camera diversity, sparse interaction modeling (both person-person and person-object), and insufficient individual attribute alignment. To bridge these gaps, we present **OmniHuman**, a large-scale, multi-scene dataset designed for fine-grained human modeling. OmniHuman provides a hierarchical annotation covering video-level scenes, frame-level interactions, and individual-level attributes. To facilitate this, we develop a fully automated pipeline for high-quality data collection and multi-modal annotation. Complementary to the dataset, we establish the OmniHuman Benchmark (**OHBench**), a three-level evaluation system that provides a scientific diagnosis for human-centric audio-video synthesis. Crucially, OHBench introduces metrics that are highly consistent with human perception, filling the gaps in existing benchmarks by providing a comprehensive diagnosis across global scenes, relational interactions, and individual attributes. Experiments show that fine-tuning on only 20% of OmniHuman significantly boosts performance, validating its effectiveness in advancing complex scenario modeling.

![Figure 1](assets/figure1_small.png)

---

<!--
## 📊 Data Pipeline

![Data Pipeline](assets/figure2_small.png)

---
-->

## 📥 Dataset

| Dataset | Hugging Face |
|---|---|
| `omnihuman_dataset` | [**Hugging Face dataset page**](https://huggingface.co/datasets/julia527/omnihuman_dataset) |

For download and usage, please follow the instructions on the Hugging Face dataset page.

---

## 📈 Benchmark distribution

![Benchmark distribution](assets/bench_distribution.png)

---

## 🧪 OHBench (benchmark + assets)

OHBench lives under `omnihuman/ohbench/`.

### OHBench assets (models + `ohbench_dir.tar`)

| Assets | Hugging Face |
|---|---|
| `models/` + `ohbench_dir.tar` | [**Hugging Face model repo**](https://huggingface.co/julia527/omnihuman_benchmark) |

After downloading, make sure you have the following **two assets under `omnihuman/ohbench/`**:

- **`omnihuman/ohbench/models/`**: all model checkpoints (mostly `.pt` / `.onnx` plus a few required model directories)
- **`omnihuman/ohbench/ohbench_dir/`**: extracted benchmark assets referenced by `ohbench/configs/paths.env`

Example (download + extract):

```bash
cd omnihuman/ohbench

# Download ONLY the `models/` folder into ./models/
huggingface-cli download julia527/omnihuman_benchmark \
  --repo-type model \
  --local-dir . \
  --include "models/**"

# Download the benchmark assets tarball, then extract (creates ./ohbench_dir/)
huggingface-cli download julia527/omnihuman_benchmark \
  --repo-type model \
  --local-dir . \
  --include "ohbench_dir.tar"

tar -xf ohbench_dir.tar
```

---

## 🛠️ Installation

Clone the repository (replace the URL when you publish):

```bash
git clone https://github.com/YOUR_ORG/omnihuman.git
cd omnihuman
```

**GPU:** install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) if you use Docker with `--gpus all`; on the host, install a CUDA-capable driver.

### Option 1: Docker (recommended)

We provide a **pre-built image** with dependencies. Replace `YOUR_ORG/ohbench-omnihuman:latest` with your published name (or a local tag).

```bash
docker pull YOUR_ORG/ohbench-omnihuman:latest

docker run -it --gpus all --network host --ipc host \
  -v /path/to/omnihuman:/workspace \
  --name ohbench \
  YOUR_ORG/ohbench-omnihuman:latest \
  bash
```

Inside the container, runs use `/workspace/ohbench` (your mounted clone, including `models/` and `ohbench_dir/`). See [Run](#run) below.

### Option 2: Conda (host)

Create a clean environment, install FFmpeg, then install PyTorch + runtime deps:

```bash
# Create environment
conda create -n ohbench python=3.10 -y
conda activate ohbench
conda install -y -c conda-forge ffmpeg

# Install PyTorch (pick the CUDA build that matches your driver/toolkit)
# Example (CUDA 12.1):
pip install -U "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" --index-url https://download.pytorch.org/whl/cu121

# Install OHBench python deps (in the cloned repo)
cd ohbench
pip install -U pip
pip install -r requirements.txt
```

---

## 🧾 OHBench metrics

Top-level categories and metric keys:

- **`video_quality`**
  - **metrics**: `IQ(imaging_quality)`, `DD(dynamic_degree)`, `IC(id_csim_single)`, `IC*(id_csim_double)`, `V-A(imagebind-av)`, `T-A(clap_score)`
  - **modules**: `evaluators/video_quality`, `evaluators/identity_consistency`, `evaluators/av_semantic_alignment`
- **`audio_quality`**
  - **metrics**: `FD`, `KL`, `AbS`, `WER`, `LSE-C`
  - **module**: `evaluators/audio_quality`
- **`speech_quality`**
  - **metrics**: `SQ` (DNSMOS overall MOS)
  - **module**: `evaluators/speech_quality`
- **`person_person`**
  - **metrics**: `IN`, `ES`, `LR` (LLM-based; double-person videos only)
  - **module**: `evaluators/person-person`

---

## 🚀 Run

### Configure benchmark assets paths

Edit **`omnihuman/ohbench/configs/paths.env`** so all paths exist on your machine (or inside the container after mounts):

- `CUSTOM_IMAGE_FOLDER` — reference images for `video_quality`
- `GT_SINGLE_DIR` — single-person GT faces for `identity_consistency` (basename-aligned with videos)
- `GT_DOUBLE_DIR` — two-person GT images for `identity_consistency`
- `AUDIO_QUALITY_BENCHMARK_DIR` — `audio_quality` assets (flat layout): per `{id}`, `{id}.json` and `{id}.wav` (GT for FD/KL)
- `AV_INPUT_CSV` — CSV for AV alignment (`path`, `text`, …)
- `EVAL_CUDA_DEVICES` — e.g. `0` or `0,1,2,3`
- `EVAL_TORCH_DEVICE` — usually `cuda:0`

### Run everything

```bash
cd omnihuman/ohbench
bash scripts/run_all.sh --input_dir /path/to/generated_mp4s --output_dir /path/to/results --name my_model
```

- **input**: folder with `.mp4` files to evaluate.
- **output**: `${name}_all.json` aggregates every metric under category keys `video_quality`, `audio_quality`, `speech_quality`, `person_person` (same metric labels); intermediate per-stage files may be removed when using `--prune`.
- **merged cache**: always **`<input_dir>_audio_video_merged`** (rebuilt each run when needed).
- **`--api_key`**: required to run `person_person`.

### Advanced (single category)

Still pass `--input_dir` and `--output_dir`:

```bash
cd omnihuman/ohbench
bash scripts/run_all.sh --input_dir /path/to/videos --output_dir /path/to/out --only video_quality
bash scripts/run_all.sh --input_dir /path/to/videos --output_dir /path/to/out --only audio_quality
bash scripts/run_all.sh --input_dir /path/to/videos --output_dir /path/to/out --only speech_quality
bash scripts/run_all.sh --input_dir /path/to/videos --output_dir /path/to/out --only person_person --api_key "$BLTCY_API_KEY"
```

