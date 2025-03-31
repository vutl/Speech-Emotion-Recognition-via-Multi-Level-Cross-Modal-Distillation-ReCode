import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, Sequential
import torchaudio.transforms as T
from tqdm import tqdm
import os
import sys
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import Config
from src.data.podcast_dataset import PodcastDataset
from src.models.teacher import TeacherModel
from src.models.student import SpeechStudent
from src.models.softalign import SoftAlign

def feature_level_loss(teacher_hidden, student_hidden, soft_align_modules, layer_weights):
    loss = 0
    # teacher_hidden is a tuple of hidden states from BERT
    # student_hidden is a dict with layer indices as keys
    
    # Map teacher layer indices to student layer indices
    # It = {6, 8, 10, 12} and Is = {3, 4, 5, 6}
    student_layer_mapping = {
        6: 3,  # teacher layer 6 -> student layer 3
        8: 4,  # teacher layer 8 -> student layer 4
        10: 5, # teacher layer 10 -> student layer 5
        12: 6  # teacher layer 12 -> student layer 6
    }
    
    for t_layer, t_feat in enumerate(teacher_hidden):
        if t_layer in soft_align_modules:
            # Get corresponding student layer
            s_layer = student_layer_mapping[t_layer]
            if s_layer in student_hidden:
                # Align student features with teacher features
                aligned_feat = soft_align_modules[t_layer](student_hidden[s_layer], t_feat)
                # Calculate MSE loss between aligned features
                loss += layer_weights[t_layer] * torch.nn.MSELoss()(aligned_feat, t_feat)
    return loss

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(model, TeacherModel):
                # For teacher model, use text data
                text = batch['text']
                label = batch['label'].to(Config.DEVICE)
                logits, _ = model(text)
            else:
                # For student model, use audio data
                audio = batch['audio'].to(Config.DEVICE)
                label = batch['label'].to(Config.DEVICE)
                logits, _ = model(audio)
            
            # Calculate loss
            loss = criterion(logits, label)
            total_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def train_teacher(teacher, train_loader, test_loader, num_epochs=5):
    """Pretrain the teacher model"""
    print("Pretraining teacher model...")
    
    # Ensure teacher model is trainable
    teacher.train()
    for param in teacher.parameters():
        param.requires_grad = True
    
    optimizer = Adam(teacher.parameters(), lr=Config.LR)
    criterion = CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(train_loader, desc=f'Teacher Epoch {epoch+1}/{num_epochs}'):
            # Process text in batches
            text = batch['text']
            label = batch['label'].to(Config.DEVICE)
            
            # Get predictions
            logits, _ = teacher(text)
            
            # Calculate loss
            loss = criterion(logits, label)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Get predictions for accuracy
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
        
        # Calculate metrics
        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate_model(teacher, test_loader, criterion)
        
        print(f'Teacher Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save checkpoint for teacher model
        if (epoch + 1) % 2 == 0:
            checkpoint_path = os.path.join('checkpoints', f'teacher_epoch_{epoch+1}.pth')
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }, checkpoint_path)
    
    # Save final teacher model
    torch.save({
        'model_state_dict': teacher.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
        'accuracy': train_acc,
        'val_loss': val_loss,
        'val_accuracy': val_acc
    }, 'teacher_final.pth')
    
    return teacher

def train_meld_mced():
    # Create data transforms with adjusted mel parameters
    transform = Sequential(
        T.Resample(orig_freq=44100, new_freq=Config.SAMPLE_RATE),
        T.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=2048,
            hop_length=512,
            n_mels=64,  # Reduced from 128 to avoid zero values
            f_min=0,
            f_max=Config.SAMPLE_RATE/2
        ),
        T.AmplitudeToDB()
    )
    
    # Create datasets
    train_dataset = PodcastDataset(
        audio_dir=Config.AUDIO_DIR,
        transcript_dir=Config.TRANSCRIPT_DIR,
        csv_file=Config.TRAIN_CSV,
        transform=transform
    )
    
    test_dataset = PodcastDataset(
        audio_dir=Config.AUDIO_DIR,
        transcript_dir=Config.TRANSCRIPT_DIR,
        csv_file=Config.TEST_CSV,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    # Initialize models
    teacher = TeacherModel(freeze=False).to(Config.DEVICE)  # Don't freeze teacher during pretraining
    student = SpeechStudent().to(Config.DEVICE)
    
    # Pretrain teacher model
    teacher = train_teacher(teacher, train_loader, test_loader)
    
    # Freeze teacher after pretraining
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    # Initialize soft alignment modules for selected teacher layers
    teacher_layer_indices = [6, 8, 10, 12]  # BERT layers to use
    teacher_dim = teacher.bert.config.hidden_size
    student_dim = 512  # Hidden dimension of student model
    layer_weights = {6: 0.1, 8: 0.1, 10: 0.2, 12: 0.3}
    
    soft_align_modules = {}
    for t_layer in teacher_layer_indices:
        soft_align_modules[t_layer] = SoftAlign(student_dim, teacher_dim).to(Config.DEVICE)
    
    # Combine parameters for optimization
    params = list(student.parameters())
    for m in soft_align_modules.values():
        params += list(m.parameters())
    
    optimizer = Adam(params, lr=Config.LR)
    criterion = CrossEntropyLoss()
    
    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        student.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS}'):
            audio = batch['audio'].to(Config.DEVICE)
            text = batch['text']
            label = batch['label'].to(Config.DEVICE)  # Changed from valence to label
            
            # Get teacher predictions and hidden states
            with torch.no_grad():
                teacher_logits, teacher_hidden = teacher(text)
            
            # Get student predictions and hidden states
            student_logits, student_hidden = student(audio)
            
            # Calculate losses
            # Distillation loss using KL divergence
            distillation_loss = torch.nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(student_logits / Config.TEMPERATURE, dim=1),
                F.softmax(teacher_logits / Config.TEMPERATURE, dim=1)
            )
            
            # Task loss using cross entropy
            task_loss = criterion(student_logits, label)
            
            # Feature-level distillation loss
            feature_loss = feature_level_loss(teacher_hidden, student_hidden, soft_align_modules, layer_weights)
            
            # Combined loss
            loss = task_loss + Config.ALPHA * (distillation_loss + feature_loss)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Get predictions for accuracy
            preds = torch.argmax(student_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
        
        # Calculate training metrics
        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate_model(student, test_loader, criterion)
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{Config.NUM_EPOCHS}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join('checkpoints', f'student_epoch_{epoch+1}.pth')
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }, checkpoint_path)
    
    # Save final model
    torch.save({
        'model_state_dict': student.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
        'accuracy': train_acc,
        'val_loss': val_loss,
        'val_accuracy': val_acc
    }, 'student_final.pth')

if __name__ == '__main__':
    train_meld_mced()
