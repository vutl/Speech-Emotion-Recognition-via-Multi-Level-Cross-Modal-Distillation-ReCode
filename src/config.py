import os
import torch

# Đường dẫn gốc đến dataset (chỉnh sửa theo cấu trúc của bạn)
BASE_DATA_ROOT = r"D:\Documents\Distill\MELD dataset\MELD-RAW\MELD.Raw"

class Config:
    # File CSV
    TRAIN_CSV = os.path.join(BASE_DATA_ROOT, "train", "train_sent_emo.csv")
    DEV_CSV   = os.path.join(BASE_DATA_ROOT, "dev_sent_emo.csv")
    TEST_CSV  = os.path.join(BASE_DATA_ROOT, "test_sent_emo.csv")
    
    # Folder chứa các file mp4
    TRAIN_MP4_DIR = os.path.join(BASE_DATA_ROOT, "train", "train_splits")
    DEV_MP4_DIR   = os.path.join(BASE_DATA_ROOT, "dev", "dev_splits_complete")
    TEST_MP4_DIR  = os.path.join(BASE_DATA_ROOT, "test", "output_repeated_splits_test")
    
    NUM_CLASSES = 7
    BATCH_SIZE = 4
    LR = 1e-4
    NUM_EPOCHS = 10
    
    TEACHER_MODEL_NAME = "bert-base-uncased"
    FREEZE_TEACHER = True
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Các tham số cho distillation
    TEMPERATURE = 2.0
    ALPHA = 0.5
