import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        # (batch, in_channel, modes), (in_channel, out_channel, modes) -> (batch, out_channel, modes)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        length = x.size(-1)
        
        # Real FFT along time dimension
        x_ft = torch.fft.rfft(x, norm='ortho')

        # Create zero output tensor for frequencies
        out_ft = torch.zeros(batchsize, self.out_channels, length // 2 + 1, device=x.device, dtype=torch.cfloat)
        
        modes = min(self.modes1, x_ft.size(-1))
        out_ft[:, :, :modes] = self.compl_mul1d(x_ft[:, :, :modes], self.weights1[:, :, :modes])

        # Inverse Real FFT back to time domain
        x = torch.fft.irfft(out_ft, n=length, norm='ortho')
        return x

class PINO_BearingClassifier(nn.Module):
    def __init__(self, num_classes=4, in_channels=3, width=64, modes=64):
        super(PINO_BearingClassifier, self).__init__()
        self.width = width
        self.fc0 = nn.Conv1d(in_channels, self.width, 1)

        self.conv0 = SpectralConv1d(self.width, self.width, modes)
        self.conv1 = SpectralConv1d(self.width, self.width, modes)
        self.conv2 = SpectralConv1d(self.width, self.width, modes)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: [batch_size, 3, 25600]
        x = self.fc0(x)
        
        x1 = self.conv0(x) + self.w0(x)
        x1 = F.gelu(x1)
        
        x2 = self.conv1(x1) + self.w1(x1)
        x2 = F.gelu(x2)

        x3 = self.conv2(x2) + self.w2(x2)
        f_hat = F.gelu(x3)  # Operator representation feature tensor [batch_size, width, 25600]

        pooled = f_hat.mean(dim=-1)  # Global average pooling over time -> [batch_size, width]
        
        h = F.gelu(self.fc1(pooled))
        logits = self.fc2(h)

        return logits, f_hat
