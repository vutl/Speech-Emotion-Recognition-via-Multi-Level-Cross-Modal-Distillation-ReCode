import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import argparse
import torch
from config import Config
from training.train import train_meld_mced

def main():
    # In ra device đang được sử dụng
    print(f"Đang sử dụng device: {Config.DEVICE}")
    if torch.cuda.is_available():
        print("CUDA đã được kích hoạt!")
    else:
        print("Sử dụng CPU.")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="Chọn mode: train hoặc eval")
    args = parser.parse_args()
    
    if args.mode == "train":
        train_meld_mced()
    elif args.mode == "eval":
        print("Chưa triển khai eval.")
    else:
        print("Unsupported mode:", args.mode)

if __name__ == "__main__":
    main()
