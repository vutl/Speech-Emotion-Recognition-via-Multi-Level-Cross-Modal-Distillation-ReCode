import os
import torch

# Base directory for the dataset
BASE_DATA_ROOT = "/home4/quanpn/interspeech2025"

class Config:
    # Audio and transcript directories
    AUDIO_DIR = os.path.join(BASE_DATA_ROOT, "Audios")
    TRANSCRIPT_DIR = os.path.join(BASE_DATA_ROOT, "Podcast-SER", "Transcripts")
    
    # CSV files for labels (in Labels directory)
    TRAIN_CSV = os.path.join(BASE_DATA_ROOT, "Podcast-SER", "Labels", "train.csv")
    TEST_CSV = os.path.join(BASE_DATA_ROOT, "Podcast-SER", "Labels", "test.csv")
    
    # Model parameters
    BATCH_SIZE = 4  
    LR = 1e-4
    NUM_EPOCHS = 50
    
    # Teacher model settings
    TEACHER_MODEL_NAME = "bert-base-uncased"
    FREEZE_TEACHER = True
    
    # Device settings
    DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    # Distillation parameters
    TEMPERATURE = 2.0
    ALPHA = 0.5
    
    # Audio processing parameters
    SAMPLE_RATE = 16000
    MAX_AUDIO_LENGTH = 10  # seconds
    
    # Training parameters
    NUM_WORKERS = 4  # Set to 0 for testing to avoid multiprocessing issues
    PIN_MEMORY = True
