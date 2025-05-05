"Clean data and create CSV manifest with checksum"
import os
import hashlib
import pandas as pd
from tqdm import tqdm

def compute_sha256(path, chunk_size=8192):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()

def is_audio_sub_broken(audio_path: str, broken_file_check: pd.DataFrame):
    return audio_path in broken_file_check["video_id"].apply(lambda x: f"video_{x}").to_list()

root_dir = "./data-local/data"
broken_file_check = pd.read_csv("./manifest/asmr-broken-file-list.csv")
manifest = []

for channel in tqdm(os.listdir(root_dir)):
    channel_path = os.path.join(root_dir, channel)
    if not os.path.isdir(channel_path): continue

    for video_folder in os.listdir(channel_path):
        video_path = os.path.join(channel_path, video_folder)
        if not os.path.isdir(video_path): continue

        files = os.listdir(video_path)
        video_id = None
        webm_file = None
        en_vtt = None
        ja_vtt = None

        for f in files:
            if f.endswith(".webm"):
                video_id = f.split(".")[0]
                webm_file = f
            elif f.endswith(".en.vtt"):
                en_vtt = f
            elif f.endswith(".ja.vtt"):
                ja_vtt = f

        if video_id and webm_file:
            audio_path = os.path.join(video_path, webm_file)
            checksum = compute_sha256(audio_path)
            video_idx_str = audio_path.split("/")[-2] # video_<int>
            manifest.append({
                "channel": channel,
                "video_id": video_id,
                "video_idx": f'{int(video_idx_str.replace("video_", "")):04}',
                "audio_path": audio_path,
                "en_subtitle_path": os.path.join(video_path, en_vtt) if en_vtt else "",
                "ja_subtitle_path": os.path.join(video_path, ja_vtt) if ja_vtt else "",
                "checksum": checksum,
                "broken_file_check":  is_audio_sub_broken(audio_path=video_idx_str, broken_file_check=broken_file_check)
            })

df = pd.DataFrame(manifest)
df.sort_values(by=["channel", "video_idx"], inplace=True)
df.to_csv("./manifest/asmr-data.csv", index=False)
print(f"Saved manifest with {len(df)} entries.")