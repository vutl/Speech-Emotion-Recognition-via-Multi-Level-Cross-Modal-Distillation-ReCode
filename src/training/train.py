import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config
from data.dataset import MELDDataset, meld_collate_fn
from models.teacher import TeacherModel
from models.student import SpeechStudent
from models.softalign import SoftAlign
from training.losses import response_level_loss, feature_level_loss
from training.eval import evaluate_model

def train_meld_mced():
    # Tạo dataset và DataLoader cho train và dev
    train_set = MELDDataset(csv_path=Config.TRAIN_CSV, mp4_folder=Config.TRAIN_MP4_DIR)
    train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=meld_collate_fn)
    
    dev_set = MELDDataset(csv_path=Config.DEV_CSV, mp4_folder=Config.DEV_MP4_DIR)
    dev_loader = DataLoader(dev_set, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=meld_collate_fn)
    
    # Khởi tạo Teacher Model (đông cứng)
    teacher_model = TeacherModel().to(Config.DEVICE)
    
    # Khởi tạo Student Model
    student_model = SpeechStudent(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    
    # Khởi tạo các module SoftAlign cho các layer teacher 8,9,10,11
    teacher_layer_indices = [8, 9, 10, 11]
    teacher_dim = teacher_model.bert.config.hidden_size  # ví dụ: 768
    student_dim = 512  # như định nghĩa trong SpeechStudent
    layer_weights = {8: 0.1, 9: 0.1, 10: 0.2, 11: 0.3}
    
    soft_align_modules = {}
    for t_layer in teacher_layer_indices:
        soft_align_modules[t_layer] = SoftAlign(student_dim, teacher_dim).to(Config.DEVICE)
    
    # Gom tất cả các tham số của student và các module soft align để tối ưu
    params = list(student_model.parameters())
    for m in soft_align_modules.values():
        params += list(m.parameters())
    
    optimizer = optim.Adam(params, lr=Config.LR)
    ce_loss = nn.CrossEntropyLoss()
    
    for epoch in range(Config.NUM_EPOCHS):
        student_model.train()
        running_loss = 0.0
        for batch in train_loader:
            audio = batch["audio"].to(Config.DEVICE)
            texts = batch["text"]
            labels = batch["label"].to(Config.DEVICE)
            
            # Forward Teacher (với text)
            teacher_logits, all_hidden = teacher_model(texts)
            teacher_feats = {}
            for l in teacher_layer_indices:
                teacher_feats[l] = all_hidden[l]
            
            # Forward Student (với audio)
            student_logits, student_hidden = student_model(audio)
            
            # Tính các thành phần loss
            loss_sup = ce_loss(student_logits, labels)
            loss_rl = response_level_loss(teacher_logits, student_logits, temperature=Config.TEMPERATURE)
            loss_fl = feature_level_loss(teacher_feats, student_hidden, soft_align_modules, layer_weights)
            
            loss_total = loss_sup + Config.ALPHA * (loss_rl + loss_fl)
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            running_loss += loss_total.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] - Loss: {avg_loss:.4f}")
        
        dev_acc = evaluate_model(student_model, dev_loader, Config.DEVICE)
        print(f"Dev Accuracy: {dev_acc:.2f}%")
    
    torch.save(student_model.state_dict(), "student_meld_mced.pth")
    print("Training completed and model saved.")
