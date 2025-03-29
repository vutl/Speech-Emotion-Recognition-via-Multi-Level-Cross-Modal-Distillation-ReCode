import os
import pandas as pd
from config import Config
from utils.audio_utils import convert_mp4_to_wav

def process_dataset(csv_path, mp4_folder):
    """
    Duyệt qua CSV và chuyển đổi các file MP4 sang WAV nếu chưa tồn tại.
    """
    df = pd.read_csv(csv_path)
    total = len(df)
    converted = 0
    failed = 0

    for idx, row in df.iterrows():
        # Giả sử CSV có cột "Dialogue_ID" và "Utterance_ID"
        dialogue_id = row["Dialogue_ID"]
        utterance_id = row["Utterance_ID"]
        mp4_filename = f"dia{dialogue_id}_utt{utterance_id}.mp4"
        mp4_path = os.path.join(mp4_folder, mp4_filename)
        wav_path = os.path.splitext(mp4_path)[0] + ".wav"
        
        # Nếu file WAV đã tồn tại, bỏ qua
        if os.path.exists(wav_path):
            continue

        print(f"Converting {mp4_path} to WAV...")
        success = convert_mp4_to_wav(mp4_path, wav_path)
        if success:
            converted += 1
        else:
            failed += 1

    print(f"Processed {total} samples. Converted: {converted}, Failed: {failed}")

if __name__ == "__main__":
    # Xử lý cho tất cả các tập dữ liệu
    print("Processing TRAIN dataset...")
    process_dataset(Config.TRAIN_CSV, Config.TRAIN_MP4_DIR)
    
    print("Processing DEV dataset...")
    process_dataset(Config.DEV_CSV, Config.DEV_MP4_DIR)
    
    print("Processing TEST dataset...")
    process_dataset(Config.TEST_CSV, Config.TEST_MP4_DIR)
    
    print("All datasets processed successfully!")
