import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import argparse
import torch
from src.config import Config
from src.training.train import train_meld_mced
from src.training.evaluate import evaluate_model

def main():
    # Print device information
    print(f"Using device: {Config.DEVICE}")
    if torch.cuda.is_available():
        print("CUDA is available!")
    else:
        print("Using CPU.")
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="Mode: train or eval")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--alpha", type=float, default=0.5, help="Distillation weight")
    parser.add_argument("--temperature", type=float, default=2.0, help="Temperature for distillation")
    parser.add_argument("--model_path", type=str, default="student_final.pth", help="Path to model for evaluation")
    args = parser.parse_args()
    
    # Update config with command line arguments
    Config.BATCH_SIZE = args.batch_size
    Config.LR = args.lr
    Config.NUM_EPOCHS = args.epochs
    Config.ALPHA = args.alpha
    Config.TEMPERATURE = args.temperature
    
    if args.mode == "train":
        print("Starting training...")
        print(f"Batch size: {Config.BATCH_SIZE}")
        print(f"Learning rate: {Config.LR}")
        print(f"Number of epochs: {Config.NUM_EPOCHS}")
        print(f"Distillation weight (alpha): {Config.ALPHA}")
        print(f"Temperature: {Config.TEMPERATURE}")
        train_meld_mced()
    elif args.mode == "eval":
        print(f"Starting evaluation with model: {args.model_path}")
        evaluate_model(args.model_path)
    else:
        print("Unsupported mode:", args.mode)

if __name__ == "__main__":
    main()
