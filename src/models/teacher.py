import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from config import Config

class TeacherModel(nn.Module):
    def __init__(self, model_name=Config.TEACHER_MODEL_NAME, freeze=Config.FREEZE_TEACHER):
        super(TeacherModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,  # For 3-class classification: negative, neutral, positive
            output_hidden_states=True
        )
        if freeze:
            self.bert.eval()
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, text_list):
        # Process text in batches
        enc = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(Config.DEVICE)
        attn_mask = enc["attention_mask"].to(Config.DEVICE)
        
        # Remove torch.no_grad() to allow gradient computation during training
        out = self.bert(input_ids=input_ids, attention_mask=attn_mask)
            
        return out.logits, out.hidden_states  # Return logits directly for classification
