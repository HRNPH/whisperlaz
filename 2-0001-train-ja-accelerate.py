#!/usr/bin/env python
# coding: utf-8

import os
import gc
import datasets
import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, IterableDataset, Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)
import evaluate
from sklearn.model_selection import train_test_split
from typing import Any, Dict, List, Union
from dataclasses import dataclass
import re
import unicodedata

@dataclass
class Config:
    OUTPUT_DIR = "./whisper-ja-asmr-tiny-2-earlyst-normalize-warm"
    MODEL_NAME = "openai/whisper-tiny"
    BATCH_SIZE = 32
    ACCUM_STEPS = 2
    RANDOM_STATE_SEED = 42
    
def flush_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("✅ GPU memory flushed.")

def normalize_japanese(text: str) -> str:
    # 1) Unicode normalize fullwidth→halfwidth, etc.
    text = unicodedata.normalize("NFKC", text)
    
    # 2) Remove any bracketed annotations like [applause], (music), etc.
    text = re.sub(r"[\[\(].*?[\]\)]", "", text)
    
    # 3) Collapse multiple spaces (sometimes subs have weird spacing)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

# Example
# raw = "これはテストです。（笑）　これは[拍手]とてもいいですね！"
# print(normalize_japanese(raw))
# # → これはテストです。 これはとてもいいですね！

def preprocess(row, processor):
    data = np.load(row["npz_path"], allow_pickle=True)
    waveform = data["audio"]
    text = str(data["text"])

    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    norm = normalize_japanese(text)
    labels = processor.tokenizer(
        norm,
        return_tensors="pt"
    ).input_ids[0]

    return {
        "input_features": inputs.input_features[0],
        "labels": labels
    }

class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor, padding=True, return_tensors="pt"):
        self.processor = processor
        self.padding = padding
        self.return_tensors = return_tensors

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features, padding=self.padding, return_tensors=self.return_tensors
        )
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": label_features}, padding=self.padding, return_tensors=self.return_tensors, return_attention_mask=True
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )
        batch["labels"] = labels
        return batch

def compute_metrics(pred, processor, wer_metric=evaluate.load("wer"), cer_metric=evaluate.load("cer")):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # ─── FILTER OUT EMPTY REFS ────────────────────
    valid = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
    if not valid:
        return {"wer": float("nan"), "cer": float("nan")}
    preds, refs = zip(*valid)
    # ────────────────────────────────────────────────
    
    return {
        "wer": wer_metric.compute(predictions=list(preds), references=list(refs)),
        "cer": cer_metric.compute(predictions=list(preds), references=list(refs))
    }

def main():
    from accelerate import Accelerator
    accelerator = Accelerator(split_batches=True)  # ✅ Fix for IterableDataset batching
    
    print(f"Accelerate is using device: {accelerator.device}")
    flush_gpu()

    os.environ["WANDB_PROJECT"] = "whisperlaz-asr-ja"
    os.environ["WANDB_LOG_MODEL"] = "false" # to collect no Artifact, or else your Wanndb Storage goes DOOM DOOM

    # Load dataset
    df = pd.read_csv("./manifest/preprocessed-segments-index.csv")
    df = df[df.lang == "ja"].reset_index(drop=True)
    print(f"Loaded {len(df)} JA training samples")

    # Split
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=Config.RANDOM_STATE_SEED)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=Config.RANDOM_STATE_SEED)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Load processor and model
    model_name = Config.MODEL_NAME
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Estimate steps
    num_samples = len(train_df)
    batch_size = Config.BATCH_SIZE
    accum_steps = Config.ACCUM_STEPS
    steps_per_epoch = num_samples // (batch_size * accum_steps)
    max_steps = steps_per_epoch * 5
    print(f"🧾 Estimated max_steps: {max_steps}")

    model.model_input_names = ["input_features"]

    # Dataset
    dataset = DatasetDict({
        "train": Dataset.from_generator(lambda: map(
            lambda r: preprocess(r, processor), train_df.to_dict(orient="records"))),
        "val": Dataset.from_generator(lambda: map(
            lambda r: preprocess(r, processor), val_df.to_dict(orient="records"))),
        "test": Dataset.from_generator(lambda: map(
            lambda r: preprocess(r, processor), test_df.to_dict(orient="records")))
    })

    # Metric
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    save_steps = 0.05

    training_args = Seq2SeqTrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accum_steps,
        max_steps=max_steps,
        learning_rate=5e-6,
        fp16=True,
        logging_steps=50,
        save_steps=save_steps, # Save every 5%
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=save_steps, # Eval every Save
        predict_with_generate=True,
        generation_max_length=50,
        eval_on_start=True,
        generation_num_beams=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False, # Minimize Wer
        report_to="wandb",
        save_only_model=False,
        save_safetensors=True,
        ddp_find_unused_parameters=False,
        warmup_ratio=0.2
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, padding=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        data_collator=data_collator,
        # tokenizer=processor,  # deprecated but still works until v5
        processing_class=processor,
        compute_metrics=lambda p: compute_metrics(p, processor, wer_metric, cer_metric),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )

    trainer.train()
    trainer.save_model(f"{Config.OUTPUT_DIR}/final")

if __name__ == "__main__":
    main()