#!/usr/bin/env python
# coding: utf-8
"""
Standalone script to merge LoRA adapters into a base Whisper model checkpoint.
Usage:
    python merge_lora.py \
        --base-model path/to/saved_model \
        --lora-adapter path/to/lora_adapter \
        --output-dir path/to/merged_model
"""
import argparse
import os
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoTokenizer
from peft import PeftModel

def merge_lora(base_model_path: str, lora_adapter_path: str, output_dir: str):
    # Load the base Whisper model
    print(f"Loading Whisper base model from '{base_model_path}'...")
    base_model = WhisperForConditionalGeneration.from_pretrained(base_model_path)

    # Apply the LoRA adapter
    print(f"Applying LoRA adapter from '{lora_adapter_path}'...")
    peft_model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    # Merge LoRA weights into the base model
    print("Merging LoRA weights into the base model...")
    merged_model = peft_model.merge_and_unload()

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save merged model weights and config
    print(f"Saving merged model state_dict to '{output_dir}/pytorch_model.bin'...")
    torch.save(merged_model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    print(f"Saving model config to '{output_dir}'...")
    merged_model.config.save_pretrained(output_dir)

    # Save tokenizer
    print(f"Saving tokenizer to '{output_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_dir)

    # Save processor to include feature-extractor and preprocessor configs
    print(f"Saving processor to '{output_dir}' (preprocessor_config.json + feature_extractor)...")
    processor = WhisperProcessor.from_pretrained(base_model_path)
    processor.save_pretrained(output_dir)

    print("✅ Merge complete. You can now load the merged model directly with WhisperForConditionalGeneration.from_pretrained without an adapter.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into a base Whisper model.")
    parser.add_argument(
        "--base-model", type=str, required=True,
        help="Path to the base Whisper model directory"
    )
    parser.add_argument(
        "--lora-adapter", type=str, required=True,
        help="Path to the LoRA adapter directory"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory where the merged model will be saved"
    )
    args = parser.parse_args()
    merge_lora(args.base_model, args.lora_adapter, args.output_dir)
