# Efficient Fine-Tuning of LLMs for Vietnamese Sentiment Analysis

Fine-tune Vistral-7B-Chat for Vietnamese sentiment classification using LoRA, 4-bit quantization, and mixed precision — with optional LLM-based data augmentation (GPT-4o-mini). This repo contains end-to-end scripts for preprocessing, training, evaluation, and inference, along with Docker support.

## Highlights

- Parameter-efficient fine-tuning (LoRA) with 4-bit loading (QLoRA-style)
- Mixed precision (FP16) for speed and memory savings  
- Optional synthetic data generation to boost generalization
- Reproducible experiments on VLSP 2016 and AIVIVN 2019

## Project Structure

```
.
├─ models/                    # Saved LoRA adapters / checkpoints
├─ output/                    # Logs, metrics, artifacts
├─ processed_data/            # Tokenized / split datasets
├─ raw_data/                  # Raw datasets (VLSP 2016, AIVIVN 2019)
├─ Dockerfile
├─ build.sh                   # Convenience build/run script for Docker
├─ evaluation.py              # Compute metrics (Accuracy, Macro F1)
├─ gen_data.py                # Synthetic data generation with GPT-4o-mini (optional)
├─ predict.py                 # Batch or single-text prediction
├─ preprocess.py              # Cleaning, splitting, instruction formatting
├─ requirements.txt
├─ test_model.py              # Test fine-tuned model
├─ test_with_gpt.py           # Baseline eval with GPT-4o (optional)
└─ training_model.py          # Fine-tuning loop (LoRA + 4-bit + FP16)
```

## Requirements

**Python:** 3.10+

**GPU:** NVIDIA GPU with CUDA (tested with RTX 4090; 12–24GB VRAM recommended)

**Drivers:** Recent CUDA toolkit & drivers compatible with your PyTorch build

Install dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Docker (optional)

```bash
# Build & run container (edit build.sh to suit your CUDA runtime)
bash build.sh
```

