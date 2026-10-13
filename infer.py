"""Generate with a saved LoRA/QLoRA adapter.

Examples:
  python infer.py --adapter outputs/qlora --prompt "Describe a stoneware bowl."
  python infer.py --adapter outputs/qlora
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", type=Path, default=Path("outputs/qlora"))
    parser.add_argument("--prompt", default=None, help="One user message. Omit this to chat in the terminal.")
    parser.add_argument("--system", default="You write short product copy in a calm, precise voice. No hype.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--load-in-4bit", action="store_true", help="Load the base model in 4-bit. Use this on a small GPU.")
    parser.add_argument("--merge-to", type=Path, default=None, help="Also save a full merged model to this folder.")
    return parser.parse_args()


def base_model_id(adapter_dir: Path) -> str:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise SystemExit(f"No adapter_config.json in {adapter_dir}. Train first with train.py.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_id = config.get("base_model_name_or_path")
    if not model_id:
        raise SystemExit(f"{config_path} has no base_model_name_or_path.")
    return model_id


def load_model(adapter_dir: Path, load_in_4bit: bool):
    model_id = base_model_id(adapter_dir)
    if torch.cuda.is_available():
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs = {"dtype": compute_dtype, "device_map": "auto"}
    else:
        compute_dtype = torch.float32
        kwargs = {"dtype": compute_dtype}
    if load_in_4bit:
        if not torch.cuda.is_available():
            raise SystemExit("4-bit loading needs a CUDA GPU.")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def generate_reply(model, tokenizer, prompt: str, system: str | None, max_new_tokens: int, temperature: float) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    do_sample = temperature > 0
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model(args.adapter, args.load_in_4bit)

    if args.merge_to is not None:
        if args.load_in_4bit:
            raise SystemExit("Merge a 16-bit model. Re-run without --load-in-4bit.")
        merged = model.merge_and_unload()
        args.merge_to.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(args.merge_to)
        tokenizer.save_pretrained(args.merge_to)
        print(f"Merged model saved to {args.merge_to}")

    if args.prompt:
        print(generate_reply(model, tokenizer, args.prompt, args.system, args.max_new_tokens, args.temperature))
        return

    print("Chat with the adapter. Empty line or 'quit' exits.")
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt or prompt.lower() in {"quit", "exit"}:
            break
        reply = generate_reply(model, tokenizer, prompt, args.system, args.max_new_tokens, args.temperature)
        print(f"\nModel: {reply}")


if __name__ == "__main__":
    main()
