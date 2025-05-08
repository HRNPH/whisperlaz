#%%
import os
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import evaluate
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from dataclasses import dataclass

@dataclass
class Config:
    RANDOM_STATE_SEED = 42
    MODEL_LIST = [
        "japanese-asr/distil-whisper-large-v3-ja-reazonspeech-all",
        "openai/whisper-tiny",
        "openai/whisper-small",
        "openai/whisper-medium",
        "openai/whisper-large-v3",
    ]

#%% 1) Load test split
df = pd.read_csv("./manifest/preprocessed-segments-index.csv")
df = df[df.lang == "ja"].reset_index(drop=True)
_, test_df = np.split(df.sample(frac=1, random_state=Config.RANDOM_STATE_SEED),
                      [int(len(df)*0.9)])
print(f"Loaded {len(test_df)} test samples")

#%% 2) Metrics & device
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#%% 3) Helper to evaluate one model
def evaluate_model(model_name: str):
    processor = WhisperProcessor.from_pretrained(model_name)
    model     = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens      = []

    preds, refs = [], []
    for _, row in tqdm(test_df.iterrows(),
                       total=len(test_df),
                       desc=f"Eval {model_name}"):
        data = np.load(row["npz_path"], allow_pickle=True)
        audio_array = data["audio"].astype(np.float32)  # raw waveform
        transcript  = str(data["text"])

        # *** CORRECTED HERE ***
        inputs = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                inputs,
                max_length=256,
                num_beams=1,
            )
        pred = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        preds.append(pred)
        refs.append(transcript)

    wer = wer_metric.compute(predictions=preds, references=refs)
    cer = cer_metric.compute(predictions=preds, references=refs)
    return wer, cer

#%% 4) Run through your models
results = {}
for m in Config.MODEL_LIST:
    wer, cer = evaluate_model(m)
    results[m] = {"wer": wer, "cer": cer}

#%% 5) Build a pandas DataFrame
df_results = pd.DataFrame([
    {"model": model_name, "wer": metrics["wer"], "cer": metrics["cer"]}
    for model_name, metrics in results.items()
])
df_results