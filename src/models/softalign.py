import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftAlign(nn.Module):
    def __init__(self, student_dim, teacher_dim):
        super(SoftAlign, self).__init__()
        self.attention = nn.MultiheadAttention(teacher_dim, num_heads=8)
        self.proj = nn.Linear(student_dim, teacher_dim)
        
    def forward(self, student_feat, teacher_feat):
        # student_feat: (batch, student_len, student_dim)
        # teacher_feat: (batch, teacher_len, teacher_dim)
        
        # Project student features to teacher dimension
        student_proj = self.proj(student_feat)
        
        # Reshape for attention
        student_proj = student_proj.transpose(0, 1)  # (student_len, batch, teacher_dim)
        teacher_feat = teacher_feat.transpose(0, 1)  # (teacher_len, batch, teacher_dim)
        
        # Apply attention with student as query and teacher as key/value
        aligned_feat, _ = self.attention(student_proj, teacher_feat, teacher_feat)
        
        # Reshape back
        aligned_feat = aligned_feat.transpose(0, 1)  # (batch, student_len, teacher_dim)
        
        # Interpolate to match teacher sequence length
        aligned_feat = F.interpolate(
            aligned_feat.transpose(1, 2),  # (batch, teacher_dim, student_len)
            size=teacher_feat.size(0),      # teacher_len
            mode='linear',
            align_corners=False
        ).transpose(1, 2)  # (batch, teacher_len, teacher_dim)
        
        return aligned_feat
