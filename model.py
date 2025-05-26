import torch.nn as nn
import torchvision.models as models
import torch


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, lstm_out):
        # lstm_out: (batch_size, seq_len, hidden_size)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)  # (batch_size, seq_len, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)       # (batch_size, hidden_size)
        return context


class CNN_LSTM(nn.Module):
    def __init__(self, num_classes, sequence_length, hidden_dim=256, lstm_layers=2, dropout=0.5, pretrained=True):
        super(CNN_LSTM, self).__init__()

        # Load pre-trained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        resnet.fc = nn.Identity()  # Remove classification head

        self.cnn = resnet
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.layernorm = nn.LayerNorm(hidden_dim * 2)
        self.attention = Attention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        # CNN feature extraction
        x = x.view(batch_size * seq_len, *x.size()[2:])
        cnn_out = self.cnn(x)  # (batch_size * seq_len, 512)
        cnn_out = cnn_out.view(batch_size, seq_len, -1)  # (batch_size, seq_len, 512)

        # LSTM
        lstm_out, _ = self.lstm(cnn_out)  # (batch_size, seq_len, hidden_dim*2)
        lstm_out = self.layernorm(lstm_out)

        # Attention + Classification
        attn_out = self.attention(lstm_out)
        out = self.fc(attn_out)
        return out