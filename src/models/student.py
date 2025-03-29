import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeechStudent(nn.Module):
    def __init__(self, input_dim=130, conv_channels=128, num_conv_layers=4, transformer_layers=6, num_heads=4, hidden_dim=512, num_classes=7, selected_layers=[2,3,4,5]):
        super(SpeechStudent, self).__init__()
        
        # Conv1D block
        conv_blocks = []
        for i in range(num_conv_layers):
            in_channels = input_dim if i == 0 else conv_channels
            conv_blocks.append(nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1))
            conv_blocks.append(nn.ReLU())
        self.conv1d = nn.Sequential(*conv_blocks)
        
        self.conv_proj = nn.Linear(conv_channels, hidden_dim)
        
        # Transformer Encoder layers
        layers = []
        for _ in range(transformer_layers):
            layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
            layers.append(layer)
        self.transformer = nn.ModuleList(layers)
        
        self.selected_layers = selected_layers
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, time, input_dim)
        x = x.permute(0, 2, 1)  # (batch, input_dim, time)
        x = self.conv1d(x)      # (batch, conv_channels, time)
        x = x.permute(0, 2, 1)   # (batch, time, conv_channels)
        
        x = self.conv_proj(x)   # (batch, time, hidden_dim)
        x = x.transpose(0, 1)   # (time, batch, hidden_dim)
        hidden_states = {}
        for i, layer in enumerate(self.transformer):
            x = layer(x)
            if i in self.selected_layers:
                hidden_states[i] = x.transpose(0, 1)  # (batch, time, hidden_dim)
        x = x.transpose(0, 1)  # (batch, time, hidden_dim)
        pooled, _ = torch.max(x, dim=1)
        logits = self.classifier(pooled)
        
        return logits, hidden_states
