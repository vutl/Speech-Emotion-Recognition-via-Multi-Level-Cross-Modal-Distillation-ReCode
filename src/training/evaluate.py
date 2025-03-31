import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, Sequential
import torchaudio.transforms as T
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from src.config import Config
from src.data.podcast_dataset import PodcastDataset
from src.models.teacher import TeacherModel
from src.models.student import SpeechStudent

def evaluate_model(model_path):
    # Create data transforms - match exactly with training configuration
    transform = Sequential(
        T.Resample(orig_freq=44100, new_freq=Config.SAMPLE_RATE),
        T.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=2048,  # Match training config
            hop_length=512,  # Match training config
            n_mels=64,  # Match training config
            f_min=0,  # Match training config
            f_max=Config.SAMPLE_RATE/2  # Match training config
        ),
        T.AmplitudeToDB()
    )
    
    # Create test dataset
    test_dataset = PodcastDataset(
        audio_dir=Config.AUDIO_DIR,
        transcript_dir=Config.TRANSCRIPT_DIR,
        csv_file=Config.TEST_CSV,
        transform=transform
    )
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    # Load model
    student = SpeechStudent().to(Config.DEVICE)
    checkpoint = torch.load(model_path)
    student.load_state_dict(checkpoint['model_state_dict'])
    student.eval()
    
    # Initialize metrics
    all_preds = []
    all_labels = []
    criterion = CrossEntropyLoss()
    total_loss = 0
    
    # Evaluate
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            audio = batch['audio'].to(Config.DEVICE)
            labels = batch['label'].to(Config.DEVICE)
            
            # Get predictions
            logits, _ = student(audio)
            
            # Calculate loss
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Get predicted class
            preds = torch.argmax(logits, dim=1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average Loss: {total_loss/len(test_loader):.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Negative', 'Neutral', 'Positive']))
    
    return accuracy 