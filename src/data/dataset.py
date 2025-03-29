import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
from utils.audio_utils import convert_mp4_to_wav

class MELDDataset(Dataset):
    def __init__(self, csv_path, mp4_folder, transform=None, emotion2idx=None):
        self.csv_path = csv_path
        self.mp4_folder = mp4_folder
        self.transform = transform
        
        self.df = pd.read_csv(self.csv_path)
        if emotion2idx is None:
            self.emotion2idx = {
                "neutral": 0, "happy": 1, "sad": 2, "anger": 3,
                "surprise": 4, "disgust": 5, "fear": 6
            }
        else:
            self.emotion2idx = emotion2idx

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["Utterance"]
        emotion_str = row["Emotion"].lower().strip()
        label = self.emotion2idx.get(emotion_str, 0)
        
        # Ví dụ: Tên file được xây dựng theo mẫu "dia{Dialogue_ID}_utt{Utterance_ID}.mp4"
        dialogue_id = row["Dialogue_ID"]
        utterance_id = row["Utterance_ID"]
        mp4_filename = f"dia{dialogue_id}_utt{utterance_id}.mp4"
        mp4_path = os.path.join(self.mp4_folder, mp4_filename)
        
        # Luôn chuyển đổi file MP4 sang WAV (nếu chưa có)
        wav_path = os.path.splitext(mp4_path)[0] + ".wav"
        if not os.path.exists(wav_path):
            print(f"Converting {mp4_path} to WAV...")
            success = convert_mp4_to_wav(mp4_path, wav_path)
            if not success:
                print(f"Conversion failed for {mp4_path}. Skipping this sample.")
                return None  # Bỏ qua mẫu này

        try:
            waveform, sample_rate = torchaudio.load(wav_path)
        except Exception as e:
            print(f"Error loading WAV file {wav_path}: {e}. Skipping sample.")
            return None
        
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=130
        )(waveform).squeeze(0).transpose(0, 1)
        
        if self.transform:
            mel_spec = self.transform(mel_spec)
        
        return {"audio": mel_spec, "text": text, "label": torch.tensor(label, dtype=torch.long)}

def meld_collate_fn(batch):
    # Loại bỏ các mẫu None
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        raise RuntimeError("No valid samples in the batch.")
    
    lengths = [item["audio"].size(0) for item in batch]
    max_len = max(lengths)
    
    audios, labels, texts = [], [], []
    for item in batch:
        audio = item["audio"]
        pad_len = max_len - audio.size(0)
        if pad_len > 0:
            pad = torch.zeros(pad_len, audio.size(1))
            audio = torch.cat([audio, pad], dim=0)
        audios.append(audio.unsqueeze(0))
        labels.append(item["label"].unsqueeze(0))
        texts.append(item["text"])
    
    audios = torch.cat(audios, dim=0)
    labels = torch.cat(labels, dim=0)
    return {"audio": audios, "label": labels, "text": texts}
