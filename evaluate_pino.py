import os
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[장치 확인] 구동 환경: {device}")

# =====================================================================
# 1. 아키텍처: 용량 확장 (Modes 128, Width 128) 및 RPM 컨디셔닝
# =====================================================================
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class PINO_BearingClassifier(nn.Module):
    # [수정 1] FNO 용량 대폭 확장 (modes=128, width=128)
    def __init__(self, num_classes=4, modes=128, width=128):
        super(PINO_BearingClassifier, self).__init__()
        self.modes = modes
        self.width = width
        
        # 입력 채널 4개: 3축(X,Y,Z) + 1채널(RPM 정규화 값)
        self.fc0 = nn.Conv1d(4, self.width, 1)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        
        self.physics_head = nn.Conv1d(self.width, 3, 1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.width, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, rpm_norm):
        rpm_channel = rpm_norm.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x_in = torch.cat([x, rpm_channel], dim=1)
        
        x_f = self.fc0(x_in)
        x_f = F.gelu(self.conv0(x_f) + self.w0(x_f))
        x_f = F.gelu(self.conv1(x_f) + self.w1(x_f))
        
        logits = self.classifier(x_f)
        f_hat = self.physics_head(x_f)
        return logits, f_hat

# =====================================================================
# 2. 물리 손실: 클래스 가중치 강력 적용 및 가우시안 소프트 타깃
# =====================================================================
class PINOLoss(nn.Module):
    def __init__(self, lambda_harmonic=0.1, sample_rate=12800):
        super(PINOLoss, self).__init__()
        # [수정 2] 극단적 클래스 불균형 페널티 부여
        weight = torch.tensor([1.0, 0.94, 8.36, 2.13]) 
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)
        self.lambda_harmonic = lambda_harmonic
        self.fs = sample_rate

    def compute_harmonic_loss(self, f_hat, rpm, label):
        B, C, N = f_hat.shape
        fr = rpm / 60.0 
        fft_mag = torch.abs(torch.fft.rfft(f_hat, dim=-1)) 
        pred_spectrum = fft_mag.mean(dim=1) 
        pred_spectrum = pred_spectrum / (pred_spectrum.sum(dim=-1, keepdim=True) + 1e-8)
        
        freqs = torch.fft.rfftfreq(N, d=1.0/self.fs).to(f_hat.device) 
        target_orders = torch.tensor([1.0, 2.0, 11.334, 13.666], device=f_hat.device) 
        
        target_f = (target_orders[label] * fr).unsqueeze(-1) 
        freqs_b = freqs.unsqueeze(0) 
        
        # 가우시안 소프트 타깃 오차 계산 (1.000 포화 방지)
        sigma = 5.0
        ideal_gaussian = torch.exp(-0.5 * ((freqs_b - target_f) / sigma) ** 2) 
        ideal_gaussian = ideal_gaussian / (ideal_gaussian.sum(dim=-1, keepdim=True) + 1e-8)
        
        loss_phy = torch.mean(torch.sum(torch.abs(pred_spectrum - ideal_gaussian), dim=-1))
        return loss_phy

    def forward(self, logits, f_hat, labels, rpms):
        loss_ce = self.ce(logits, labels)
        loss_h = self.compute_harmonic_loss(f_hat, rpms, labels)
        return loss_ce + self.lambda_harmonic * loss_h, loss_ce, loss_h

# =====================================================================
# 3. 데이터 로더
# =====================================================================
SIGNAL_CACHE = {}

class FastSpindleDataset(Dataset):
    def __init__(self, h5_path, meta_df, rpms=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]):
        self.h5_path = h5_path
        self.meta = meta_df.reset_index(drop=True)
        self.rpms = rpms
        
        self.samples = []
        for s_idx, row in self.meta.iterrows():
            lbl = row['label_idx']
            folder_id = row['sample_id']
            for r_idx, rpm in enumerate(self.rpms):
                self.samples.append((folder_id, r_idx, rpm, lbl))
                
        global SIGNAL_CACHE
        if len(SIGNAL_CACHE) == 0 and os.path.exists(h5_path):
            print("[진행] 신호 데이터 초고속 RAM 메모리 캐싱 중...", flush=True)
            offset = int(4.8 * 25600)
            with h5py.File(h5_path, 'r') as h5_file:
                for fid in h5_file.keys():
                    mat = h5_file[fid][:] 
                    for r_i in range(mat.shape[0]):
                        sig_w = mat[r_i, :, offset:offset + 25600:2] # Decimation (12800 pts)
                        m = np.mean(sig_w, axis=1, keepdims=True)
                        s = np.std(sig_w, axis=1, keepdims=True) + 1e-8
                        SIGNAL_CACHE[(fid, r_i)] = torch.tensor((sig_w - m) / s, dtype=torch.float32)
            print(f"[완료] 총 {len(SIGNAL_CACHE)}개 샘플 RAM 캐싱 완료!", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_id, r_idx, rpm, label = self.samples[idx]
        sig_tensor = SIGNAL_CACHE[(folder_id, r_idx)]
        rpm_norm = rpm / 8000.0
        
        return (sig_tensor, 
                torch.tensor([rpm_norm], dtype=torch.float32), 
                torch.tensor(rpm, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long),
                folder_id)

# =====================================================================
# 4. 훈련 및 검증: Cosine Scheduler 및 추론 보정(Calibration)
# =====================================================================
def evaluate_pino_kfold(df_meta, h5_path):
    df = df_meta[df_meta['label'].notna()].copy()
    label_map = {lbl: i for i, lbl in enumerate(sorted(df['label'].unique()))}
    inv_label_map = {i: lbl for lbl, i in label_map.items()}
    df['label_idx'] = df['label'].map(label_map)
    
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_folder_probs = {}
    oof_folder_true = {}
    
    # [수정 3] Epoch 수 증대
    NUM_EPOCHS = 40
    
    fold = 1
    for train_idx, val_idx in sgkf.split(df, df['label_idx'], groups=df['sample_id']):
        print(f"\n{'='*20} Fold {fold} 시작 {'='*20}", flush=True)
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        train_ds = FastSpindleDataset(h5_path, train_df)
        val_ds = FastSpindleDataset(h5_path, val_df)
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
        
        model = PINO_BearingClassifier(num_classes=len(label_map)).to(device)
        criterion = PINOLoss(lambda_harmonic=0.1, sample_rate=12800).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # [수정 4] Cosine Learning Rate Scheduler 도입
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
        
        model.train()
        for epoch in range(1, NUM_EPOCHS + 1):
            total_loss, ce_loss_acc, h_loss_acc = 0.0, 0.0, 0.0
            for x_batch, rpm_norm, rpm_real, y_batch, _ in train_loader:
                x_batch, rpm_norm = x_batch.to(device), rpm_norm.to(device)
                rpm_real, y_batch = rpm_real.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                logits, f_hat = model(x_batch, rpm_norm)
                loss, l_ce, l_h = criterion(logits, f_hat, y_batch, rpm_real)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                ce_loss_acc += l_ce.item()
                h_loss_acc += l_h.item()
                
            scheduler.step()
                
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch [{epoch}/{NUM_EPOCHS}] Loss: {total_loss/len(train_loader):.4f} (CE: {ce_loss_acc/len(train_loader):.4f}, Phy: {h_loss_acc/len(train_loader):.4f})", flush=True)
        
        model.eval()
        with torch.no_grad():
            for x_batch, rpm_norm, rpm_real, y_batch, f_ids in val_loader:
                x_batch, rpm_norm = x_batch.to(device), rpm_norm.to(device)
                
                logits, _ = model(x_batch, rpm_norm)
                
                # Temperature Scaling 기반 Softmax (과확신 방지)
                temperature = 1.5
                probs = F.softmax(logits / temperature, dim=1).cpu().numpy()
                
                for p, y_val, fid in zip(probs, y_batch.numpy(), f_ids):
                    if fid not in oof_folder_probs:
                        oof_folder_probs[fid] = []
                        oof_folder_true[fid] = y_val
                    oof_folder_probs[fid].append(p)
        fold += 1

    # [수정 5] 추론 시 소수 클래스 확률 보정 (Prior Calibration)
    y_true_all = []
    y_pred_all = []
    
    # 클래스 C, D에 대한 추가 보정 가중치 (필요 시 조절 가능)
    calibration_weights = np.array([1.0, 1.0, 2.5, 1.8])
    
    for fid, prob_list in oof_folder_probs.items():
        avg_prob = np.mean(prob_list, axis=0)
        calibrated_prob = avg_prob * calibration_weights
        pred_lbl = np.argmax(calibrated_prob)
        
        y_pred_all.append(pred_lbl)
        y_true_all.append(oof_folder_true[fid])
        
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    
    acc = accuracy_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all, average='macro')
    
    print("\n" + "="*50)
    print("   ★ [PINO 모델 5-Fold Cross-Validation Soft Voting 결과] ★")
    print("="*50)
    print(f"전체 Accuracy: {acc*100:.2f}% | Macro F1: {f1:.4f}")
    print("\n[상세 분류 보고서]")
    target_names = [inv_label_map[i] for i in range(len(label_map))]
    print(classification_report(y_true_all, y_pred_all, target_names=target_names))
    print("[혼동 행렬]")
    cm = confusion_matrix(y_true_all, y_pred_all)
    print(pd.DataFrame(cm, index=[f"실제_{c}" for c in target_names], columns=[f"예측_{c}" for c in target_names]))
    return acc, f1

if __name__ == "__main__":
    META_CSV = "./data/meta/dataset_metadata.csv"
    H5_PATH = "./data/processed/sliced_signals.h5"
    if os.path.exists(META_CSV) and os.path.exists(H5_PATH):
        df_m = pd.read_csv(META_CSV)
        evaluate_pino_kfold(df_m, H5_PATH)
    else:
        print("데이터셋 파일이 존재하지 않습니다.")