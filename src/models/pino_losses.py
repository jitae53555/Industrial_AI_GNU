import torch
import torch.nn as nn
import torch.nn.functional as F

class PINOLoss(nn.Module):
    """
    Physics-Informed Loss for PINO (Physics-Informed Neural Operator):
    Combines Cross-Entropy Data Loss with Gaussian Soft-Target Physics Spectrum Loss.
    """
    def __init__(self, lambda_harmonic=0.05, sample_rate=25600):
        super(PINOLoss, self).__init__()
        self.lambda_harmonic = lambda_harmonic
        self.sample_rate = sample_rate
        self.ce = nn.CrossEntropyLoss()

    def compute_harmonic_loss(self, f_hat, rpm, label):
        # f_hat: (B, C, T)
        fr = rpm / 60.0  # (B,)
        
        # Real FFT magnitude spectrum along time dimension
        fft_mag = torch.abs(torch.fft.rfft(f_hat, dim=-1))  # (B, C, F_bins)
        freqs = torch.fft.rfftfreq(f_hat.size(-1), d=1.0 / self.sample_rate).to(f_hat.device)  # (F_bins,)
        
        target_orders = {
            0: 1.0,      # Class A (정상): 1X 중심
            1: 2.0,      # Class B: 2X 정렬 불량 중심
            2: 11.334,   # Class C: BPFO 외륜 손상
            3: 13.666    # Class D: BPFI 내륜 손상
        }
        
        loss_phy = 0.0
        batch_size = f_hat.size(0)
        for b in range(batch_size):
            lbl_idx = int(label[b].item())
            order = target_orders.get(lbl_idx, 1.0)
            target_f = order * fr[b]
            
            # Target frequency Gaussian window (soft target)
            sigma = 5.0  # 5 Hz band
            ideal_gaussian = torch.exp(-0.5 * ((freqs - target_f) / sigma) ** 2)
            ideal_gaussian = ideal_gaussian / (ideal_gaussian.sum() + 1e-8)
            
            pred_spectrum = fft_mag[b].mean(dim=0)
            pred_spectrum = pred_spectrum / (pred_spectrum.sum() + 1e-8)
            
            # Smooth spectral distance (L1 Loss) between predicted spectrum and physical peak
            loss_phy += torch.sum(torch.abs(pred_spectrum - ideal_gaussian))
            
        return loss_phy / batch_size

    def forward(self, logits, f_hat, y_batch, rpm_batch):
        # 1. Classification Data Loss
        l_ce = self.ce(logits, y_batch)

        # 2. Differentiable Physics-Informed Harmonic Spectrum Loss
        l_h = self.compute_harmonic_loss(f_hat, rpm_batch, y_batch)
        
        total_loss = l_ce + self.lambda_harmonic * l_h
        return total_loss, l_ce, l_h
