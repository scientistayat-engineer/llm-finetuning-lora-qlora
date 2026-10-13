"""Fine-tune a causal LLM with LoRA or QLoRA.

LoRA trains a small adapter on top of a frozen base model.
QLoRA does the same thing while the base model stays in 4-bit, so a
7B model can fit on a single consumer GPU.

The dataset is supervised: each row is a conversation, and the loss is
applied only to the assistant replies.

Examples:
  python train.py --method qlora --train-file data/sample_train.jsonl
  python train.py --method lora --model Qwen/Qwen2.5-1.5B-Instruct --epochs 3
  python train.py --dataset tatsu-lab/alpaca --dataset-split train[:1000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


# Attention and MLP projections used by Qwen, Llama, Mistral, and Gemma.
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=["lora", "qlora"], default="qlora", help="qlora = 4-bit base model. lora = 16-bit base model.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face model id or local path.")
    parser.add_argument("--train-file", type=Path, default=Path("data/sample_train.jsonl"), help="Local .json or .jsonl file. Ignored when --dataset is set.")
    parser.add_argument("--dataset", default=None, help="Hugging Face dataset id. Replaces --train-file.")
    parser.add_argument("--dataset-split", default="train", help="Split to load when --dataset is set. Example: train[:2000]")
    parser.add_argument("--eval-file", type=Path, default=None, help="Optional local eval file in the same format as the training file.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where the adapter is saved. Default: outputs/<method>")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop after this many steps. -1 trains for --epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size.")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps. Effective batch = batch-size * grad-accum.")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank. Higher uses more memory and can fit the data more tightly.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA scale. A common start is 2x the rank.")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_raw_dataset(path: Path | None, dataset_id: str | None, split: str) -> Dataset:
    if dataset_id:
        dataset = load_dataset(dataset_id, split=split)
    else:
        if path is None or not path.exists():
            raise SystemExit(f"Training file not found: {path}")
        dataset = load_dataset("json", data_files=str(path), split="train")
    if not isinstance(dataset, Dataset):
        raise SystemExit("Expected a single dataset split, not a dataset dict.")
    return dataset


def convert_example(example: dict) -> list[dict]:
    """Turn one raw row into prompt/completion pairs.

    Loss is computed on the completion only. A multi-turn conversation
    becomes one pair per assistant reply, with earlier turns as the prompt.
    """
    if example.get("messages"):
        messages = example["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)
        rows = []
        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            prompt = messages[:index]
            content = str(message.get("content") or "").strip()
            if not prompt or not content:
                continue
            rows.append({"prompt": prompt, "completion": [message]})
        if not rows:
            raise ValueError("no assistant reply to learn from")
        return rows

    prompt = example.get("prompt")
    completion = example.get("completion")
    if prompt is not None and completion is not None:
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        if isinstance(completion, str):
            completion = [{"role": "assistant", "content": completion}]
        if not prompt or not completion:
            raise ValueError("empty prompt or completion")
        return [{"prompt": prompt, "completion": completion}]

    instruction = example.get("instruction")
    output = example.get("output") or example.get("response")
    if instruction and output and str(output).strip():
        user_text = str(instruction)
        extra = example.get("input")
        if extra and str(extra).strip():
            user_text = f"{user_text}\n\n{extra}"
        prompt_messages = []
        system = example.get("system")
        if system and str(system).strip():
            prompt_messages.append({"role": "system", "content": str(system)})
        prompt_messages.append({"role": "user", "content": user_text})
        return [{
            "prompt": prompt_messages,
            "completion": [{"role": "assistant", "content": str(output)}],
        }]

    raise ValueError("expected messages, prompt/completion, or instruction/output")


def to_prompt_completion(dataset: Dataset) -> Dataset:
    rows: list[dict] = []
    for index, example in enumerate(dataset):
        try:
            rows.extend(convert_example(example))
        except ValueError as exc:
            raise SystemExit(f"Row {index} cannot be used for training: {exc}") from exc
    if not rows:
        raise SystemExit("The dataset produced no training rows.")
    return Dataset.from_list(rows)


def build_quantization(compute_dtype: torch.dtype) -> BitsAndBytesConfig:
    # nf4 + double quant is the QLoRA setup from the paper: the frozen
    # base weights stay in 4-bit, and the adapter trains in compute_dtype.
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_trainer(args: argparse.Namespace, train_dataset: Dataset, eval_dataset: Dataset | None) -> SFTTrainer:
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU is visible. LoRA and QLoRA training in this script need an NVIDIA GPU.")

    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    output_dir = args.output_dir or Path("outputs") / args.method

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=DEFAULT_TARGET_MODULES,
    )

    quantization_config = build_quantization(compute_dtype) if args.method == "qlora" else None
    model_init_kwargs = {
        "attn_implementation": "sdpa",
        "dtype": compute_dtype,
        # Cache is incompatible with gradient checkpointing during training.
        "use_cache": False,
    }
    if args.method == "qlora":
        model_init_kwargs["device_map"] = "auto"

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        # 8-bit optimizer keeps QLoRA's optimizer state small.
        optim="paged_adamw_8bit" if args.method == "qlora" else "adamw_torch_fused",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        max_length=args.max_length,
        completion_only_loss=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.logging_steps if eval_dataset is not None else None,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        model_init_kwargs=model_init_kwargs,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    return SFTTrainer(
        model=args.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        quantization_config=quantization_config,
    )


def main() -> None:
    args = parse_args()
    train_raw = load_raw_dataset(args.train_file, args.dataset, args.dataset_split)
    train_dataset = to_prompt_completion(train_raw)
    eval_dataset = to_prompt_completion(load_raw_dataset(args.eval_file, None, "train")) if args.eval_file else None

    print(f"Method: {args.method}")
    print(f"Model: {args.model}")
    print(f"Training rows: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Eval rows: {len(eval_dataset)}")

    trainer = build_trainer(args, train_dataset, eval_dataset)
    trainer.train()
    trainer.save_model(trainer.args.output_dir)
    tokenizer = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.save_pretrained(trainer.args.output_dir)
    print(f"Adapter saved to {trainer.args.output_dir}")


if __name__ == "__main__":
    main()
