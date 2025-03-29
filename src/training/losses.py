import torch
import torch.nn as nn
import torch.nn.functional as F

def response_level_loss(teacher_logits, student_logits, temperature=2.0):
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean')
    return kl * (temperature ** 2)

def feature_level_loss(teacher_feats, student_feats, soft_align_modules, layer_weights):
    mse = nn.MSELoss()
    total = 0.0
    # Mapping: teacher layers [8,9,10,11] -> student layers [2,3,4,5]
    mapping = {8: 2, 9: 3, 10: 4, 11: 5}
    for t_layer, s_layer in mapping.items():
        teacher_tensor = teacher_feats[t_layer]
        student_tensor = student_feats[s_layer]
        aligned_student = soft_align_modules[t_layer](teacher_tensor, student_tensor)
        total += layer_weights[t_layer] * mse(aligned_student, teacher_tensor)
    return total
