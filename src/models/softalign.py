import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftAlign(nn.Module):
    def __init__(self, student_dim, teacher_dim):
        super(SoftAlign, self).__init__()
        self.Wq = nn.Linear(teacher_dim, teacher_dim)
        self.Wk = nn.Linear(student_dim, teacher_dim)
        self.Wv = nn.Linear(student_dim, teacher_dim)
    
    def forward(self, teacher_feat, student_feat):
        query = self.Wq(teacher_feat)
        key = F.gelu(self.Wk(student_feat))
        value = F.gelu(self.Wv(student_feat))
        d_k = teacher_feat.size(-1)
        scores = torch.matmul(query, key.transpose(1, 2)) / (d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        aligned = torch.matmul(attn_weights, value)
        return aligned
