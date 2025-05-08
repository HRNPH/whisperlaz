# %%
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
    MANIFEST_CSV = "./manifest/preprocessed-segments-index.csv"
    TEST_RATIO   = 0.1
    SEED         = 42
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
    # list your fine-tuned model and any baselines you want to compare
    MODEL_LIST   = [
        "openai/whisper-tiny",
        "openai/whisper-small",
        "openai/whisper-medium",
        "openai/whisper-large-v3",
        "./whisper-ja-asmr-tiny-1-earlyst/final"
    ]
    # these hyperparameters will be *identical* across every model:
    GENERATION_KWARGS = {
        "num_beams":            3,
        "no_repeat_ngram_size": 2,
        "repetition_penalty":   1.5,
        "length_penalty":       1.0,
        "early_stopping":       True,
        "max_new_tokens":       50,
    }

# 1) load & split test set
df = pd.read_csv(Config.MANIFEST_CSV)
df = df[df.lang == "ja"].reset_index(drop=True)
_, test_df = np.split(
    df.sample(frac=1, random_state=Config.SEED),
    [ int(len(df)*(1 - Config.TEST_RATIO)) ]
)
print(f"▶ Test examples: {len(test_df)}")

# 2) prepare metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# 3) evaluation function
def evaluate_model(model_name: str):
    print(f"\n→ Evaluating: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(Config.DEVICE)

    # **KEY FIX**: clear the *generation_config*’s forced_decoder_ids & suppress_tokens
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens      = []

    # apply your shared hyperparameters
    for k, v in Config.GENERATION_KWARGS.items():
        setattr(model.generation_config, k, v)

    preds, refs = [], []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="  samples"):
        data        = np.load(row["npz_path"], allow_pickle=True)
        audio_arr   = data["audio"].astype(np.float32)
        reference   = str(data["text"])

        # tokenize
        inputs = processor(
            audio_arr,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True
        )
        input_feats    = inputs.input_features.to(Config.DEVICE)
        attention_mask = inputs.attention_mask.to(Config.DEVICE)

        # generate with *only* generation_config
        with torch.no_grad():
            gen_ids = model.generate(
                input_feats,
                attention_mask=attention_mask,
                # no need to re-pass the hyperparams here
            )

        prediction = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        preds.append(prediction)
        refs.append(reference)

    wer = wer_metric.compute(predictions=preds, references=refs)
    cer = cer_metric.compute(predictions=preds, references=refs)
    print(f"    WER: {wer:.3f}   CER: {cer:.3f}")
    return {"wer": wer, "cer": cer}

# 4) loop through all models
results = {}
for model_name in Config.MODEL_LIST:
    results[model_name] = evaluate_model(model_name)

# 5) summary
df_results = pd.DataFrame([
    {"model": m, **metrics}
    for m, metrics in results.items()
])
print("\n### Comparison")
print(df_results)
df_results.to_csv("./model-comparison-evals.csv", index=False)