# LoRA & QLoRA Fine-Tuning of a Chat LLM

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Transformers">
  <img src="https://img.shields.io/badge/PEFT-LoRA-8A2BE2?style=for-the-badge" alt="PEFT">
  <img src="https://img.shields.io/badge/QLoRA-4--bit%20NF4-00A86B?style=for-the-badge" alt="QLoRA">
  <img src="https://img.shields.io/badge/NVIDIA-GPU%20Required-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="NVIDIA GPU">
</p>

Fine-tune a small chat language model on your own conversations by training a tiny **adapter** while the original model weights stay **frozen**. The project supports two modes:

| Mode | Base model stored as | Trains | Best when |
| --- | --- | --- | --- |
| **LoRA** | 16-bit | Adapter only | The 16-bit model fits in your GPU memory |
| **QLoRA** | 4-bit (NF4) | Adapter only | GPU memory is tight (e.g. a 7B model on one consumer GPU) |

The included sample dataset teaches the model a toy style: **short, calm, precise product copy with no hype**.

---

## 📑 Table of Contents

1. [✨ Features](#-features)
2. [📁 Project Structure](#-project-structure)
3. [🧠 How It Works](#-how-it-works)
4. [📋 Requirements](#-requirements)
5. [⚙️ Setup](#-setup)
6. [🚀 Training](#-training)
7. [🗂️ Data Format](#-data-format)
8. [💬 Inference & Merging](#-inference--merging)
9. [🎛️ Hyperparameters](#-hyperparameters)
10. [🛠️ Troubleshooting](#-troubleshooting)
11. [🎤 Presentation](#-presentation)
12. [🧰 Tech Stack](#-tech-stack)

---

## ✨ Features

- One script (`train.py`) for both **LoRA** and **QLoRA** (`--method lora|qlora`)
- **Supervised fine-tuning** with loss computed on **assistant replies only**
- Accepts three data styles: `messages`, `prompt/completion`, and `instruction/output` (Alpaca-style)
- Load data from a **local JSONL file** or directly from the **Hugging Face Hub**
- Optional **eval file** for validation loss during training
- Multi-turn conversations are split into one training example per assistant reply
- `infer.py` to **chat with the adapter** in the terminal, run a single prompt, or **merge the adapter** into a full model
- Automatic **bf16 / fp16** selection depending on the GPU

---

## 📁 Project Structure

```
llm-finetuning-lora-qlora/
├── data/
│   └── sample_train.jsonl        # Toy dataset (10 product-copy conversations)
├── presentation/
│   ├── fine-tuning-lora-qlora.html   # Briefing deck (source)
│   └── fine-tuning-lora-qlora.pdf    # Briefing deck (PDF)
├── train.py                      # LoRA / QLoRA training script
├── infer.py                      # Generate text, chat, and merge adapter
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

> Training output (the saved adapter) is written to `outputs/` and is git-ignored.

---

## 🧠 How It Works

**Fine-tuning** continues training a pretrained model on examples that look like the job you want it to do.

**LoRA (Low-Rank Adaptation)** leaves each large weight matrix `W` frozen and trains two thin matrices `A` and `B` beside it. Their product stands in for the change to that layer:

```
output = W·x + B·A·x
```

For a 1024 × 1024 layer with rank `r = 16`, `W` has 1,048,576 values while `A` and `B` together have only 32,768, about **3%** of the layer.

**QLoRA** is the same adapter, but the frozen `W` is stored in **4-bit NormalFloat (NF4)** with double quantization. Only the stored base weights shrink; the adapter still trains in 16-bit.

| Storage for the 1024 × 1024 example | Bits per value | Size |
| --- | --- | --- |
| LoRA base (16-bit) | 16 | ~2.0 MB |
| QLoRA base (NF4) | 4 | ~0.5 MB |
| Adapter `A` + `B` (either method) | 16 | ~64 KB |

On a full 7B model the same ratio applies: about 14 GB in 16-bit versus about 4 GB in 4-bit, before activations and optimizer state.

### 🔧 Training choices made in `train.py`

- **LoRA target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (works for Qwen, Llama, Mistral, Gemma)
- **QLoRA config:** NF4 + double quantization, compute dtype bf16 (or fp16 if bf16 is unsupported)
- **Optimizer:** `paged_adamw_8bit` for QLoRA, `adamw_torch_fused` for LoRA
- **Scheduler:** cosine with 3% warmup
- **Loss:** `completion_only_loss=True`, so only the assistant text is trained; a system message stays in the prompt
- **Memory savers:** gradient checkpointing and gradient accumulation
- **Trainer:** `SFTTrainer` from the `trl` library

---

## 📋 Requirements

- **NVIDIA GPU** with CUDA (CPU-only PyTorch cannot run this training script)
- Python 3.10+
- A CUDA build of PyTorch that matches your driver
- Python packages listed in `requirements.txt`:

| Package | Minimum version |
| --- | --- |
| transformers | 4.51 |
| datasets | 3.2 |
| accelerate | 1.4 |
| peft | 0.15 |
| trl | 0.16 |
| bitsandbytes | 0.45 |

---

## ⚙️ Setup

**1. Clone the repository**

```bash
git clone https://github.com/scientistayat-engineer/llm-finetuning-lora-qlora.git
cd llm-finetuning-lora-qlora
```

**2. Create a virtual environment**

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install PyTorch (CUDA build), then the rest**

Pick the exact command for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/). Example for CUDA 12.8:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Change `cu128` if your driver uses a different CUDA build.

---

## 🚀 Training

The default model is `Qwen/Qwen2.5-0.5B-Instruct`, small enough to try first.

**QLoRA on the sample data (default):**

```bash
python train.py --method qlora --train-file data/sample_train.jsonl --epochs 3
```

**LoRA on the same data (16-bit base):**

```bash
python train.py --method lora --train-file data/sample_train.jsonl
```

**A larger model, stopped after 100 steps:**

```bash
python train.py --method qlora --model Qwen/Qwen2.5-7B-Instruct --max-steps 100
```

**A dataset from the Hugging Face Hub (Alpaca-style):**

```bash
python train.py --method qlora --dataset tatsu-lab/alpaca --dataset-split "train[:1000]"
```

**With an evaluation file:**

```bash
python train.py --method qlora --train-file data/sample_train.jsonl --eval-file data/eval.jsonl
```

The adapter is saved to `outputs/qlora` or `outputs/lora` (override with `--output-dir`).

---

## 🗂️ Data Format

JSONL with one conversation per line. Loss is applied to **assistant text only**.

```json
{"messages": [
  {"role": "system", "content": "You write short product copy in a calm, precise voice. No hype."},
  {"role": "user", "content": "Describe a stoneware bowl."},
  {"role": "assistant", "content": "A stoneware bowl with a thick rim and a pale glaze."}
]}
```

(In the file each record sits on a single line.)

Also accepted:

| Style | Fields |
| --- | --- |
| Prompt / completion | `prompt`, `completion` |
| Instruction / output | `instruction`, `output` (or `response`), optional `input`, optional `system` |

Multi-turn chats become one training row per assistant reply, with earlier turns used as the prompt.

To use your own data, replace `data/sample_train.jsonl` or pass `--train-file path/to/your.jsonl`.

---

## 💬 Inference & Merging

**Single prompt (adapter on a 4-bit base model):**

```bash
python infer.py --adapter outputs/qlora --load-in-4bit --prompt "Describe a stoneware bowl."
```

**Interactive chat:** leave off `--prompt`. Type an empty line or `quit` to exit.

```bash
python infer.py --adapter outputs/qlora --load-in-4bit
```

**Merge the adapter into a full model:** needs a 16-bit load, so more VRAM and **no** `--load-in-4bit`.

```bash
python infer.py --adapter outputs/qlora --merge-to outputs/merged --prompt "Describe a stoneware bowl."
```

`infer.py` options:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--adapter` | `outputs/qlora` | Folder with the trained adapter |
| `--prompt` | none | One user message; omit for chat mode |
| `--system` | calm product-copy prompt | System message |
| `--max-new-tokens` | 200 | Length of the reply |
| `--temperature` | 0.7 | Sampling temperature (0 = greedy) |
| `--load-in-4bit` | off | Load the base model in 4-bit |
| `--merge-to` | none | Save a merged full model to this folder |

---

## 🎛️ Hyperparameters

| Flag | Default | What it does |
| --- | --- | --- |
| `--method` | `qlora` | `lora` = 16-bit base, `qlora` = 4-bit base |
| `--model` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face model id or local path |
| `--epochs` | 1.0 | Passes over the data |
| `--max-steps` | -1 | Stop after N steps (-1 = use epochs) |
| `--learning-rate` | 2e-4 | Common LoRA/QLoRA rate; full fine-tuning uses much smaller |
| `--batch-size` | 1 | Per-device batch size |
| `--grad-accum` | 8 | Effective batch = batch-size × grad-accum |
| `--max-length` | 1024 | Tokens per example; longer uses more memory |
| `--lora-r` | 16 | Adapter rank; 8–16 is a usual start, higher fits tighter but uses more memory |
| `--lora-alpha` | 32 | Adapter scale; start near 2 × rank |
| `--lora-dropout` | 0.05 | Dropout inside the adapter |
| `--logging-steps` | 10 | Log every N steps |
| `--save-steps` | 50 | Checkpoint every N steps (keeps the last 2) |
| `--seed` | 42 | Random seed |

Run `python train.py --help` for the full list.

---

## 🛠️ Troubleshooting

| Problem | Fix |
| --- | --- |
| `No CUDA GPU is visible` | Install a CUDA build of PyTorch and check `nvidia-smi`. Without a local NVIDIA GPU, run on a cloud GPU such as a Colab or Kaggle notebook. |
| CUDA out of memory | Use `--method qlora`, lower `--max-length`, keep `--batch-size 1`, raise `--grad-accum`, or pick a smaller model. |
| `Training file not found` | Check the path passed to `--train-file`. |
| `Row N cannot be used for training` | That row doesn't match a supported format or has no assistant reply. See [🗂️ Data Format](#-data-format). |
| `No adapter_config.json` in `infer.py` | Train first, or point `--adapter` at the correct output folder. |
| `Merge a 16-bit model` | Remove `--load-in-4bit` when using `--merge-to`. |
| bitsandbytes errors on Windows | Make sure you use a recent `bitsandbytes` (≥ 0.45) and a matching CUDA PyTorch build. |

---

## 🎤 Presentation

A short briefing deck explaining fine-tuning, LoRA, and QLoRA with worked examples is included:

- [`presentation/fine-tuning-lora-qlora.pdf`](presentation/fine-tuning-lora-qlora.pdf)
- [`presentation/fine-tuning-lora-qlora.html`](presentation/fine-tuning-lora-qlora.html)

---

## 🧰 Tech Stack

- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft) for LoRA adapters
- [TRL](https://github.com/huggingface/trl) for `SFTTrainer`
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) for 4-bit quantization
- [Datasets](https://github.com/huggingface/datasets) and [Accelerate](https://github.com/huggingface/accelerate)

---

## 🙏 Acknowledgements

Built as a class project at SMIT (AI and Data Science) on fine-tuning with LoRA and QLoRA.
