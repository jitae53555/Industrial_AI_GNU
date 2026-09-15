import os
import glob
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft

# --- 1. Parameters & Configuration ---
FS = 25600  # 25.6 kHz
SKIP_PTS = 122880  # 4.8s * 25600 Hz
MAX_PTS = 153600   # 6.0s * 25600 Hz

# Kinematic Characteristic Frequencies @ 8000 RPM (fr = 133.33 Hz)
FR_1X = 8000.0 / 60.0    # 133.33 Hz
FR_2X = 2.0 * FR_1X       # 266.67 Hz
BPFO_FREQ = 11.334 * FR_1X  # ~1511.2 Hz
BPFI_FREQ = 13.666 * FR_1X  # ~1822.1 Hz

META_PATH = "./data/meta/dataset_metadata.csv"
ALT_META_PATH = "./data/meta/dataset_metadata_v2.csv"
RAW_ROOT = "./data/raw"
OUTPUT_DIR = "./blind_test_charts"

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(42)  # For reproducible sample selection

# --- 2. Signal Loader ---
def load_z8000_signal(s_path):
    """Load Z-axis 8000 RPM steady-state signal."""
    pattern = os.path.join(s_path, "Raw_Z_8000*")
    cand = glob.glob(pattern)
    if not cand:
        return None

    fpath = cand[0]
    try:
        df = pd.read_csv(fpath, skiprows=SKIP_PTS, nrows=MAX_PTS, header=None, engine='c', dtype=np.float32)
        vals = df.values.ravel()
        if len(vals) == 0:
            df = pd.read_csv(fpath, nrows=MAX_PTS, header=None, engine='c', dtype=np.float32)
            vals = df.values.ravel()
        return vals
    except Exception:
        return None

# --- 3. Main Chart Generation ---
def main():
    print("=" * 60)
    print(" [Step 1] Loading Dataset & Selecting Blind Samples")
    print("=" * 60)

    meta_file = META_PATH if os.path.exists(META_PATH) else ALT_META_PATH
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"Metadata file not found at {META_PATH} or {ALT_META_PATH}")

    df_meta = pd.read_csv(meta_file)
    label_col = 'target_label' if 'target_label' in df_meta.columns else 'label'
    df_meta['label_norm'] = df_meta[label_col].astype(str).str.strip().str.upper().str[0]

    # Select 2 Class A (OK) and 2 Class D (NG) samples
    df_a = df_meta[df_meta['label_norm'] == 'A'].copy()
    df_d = df_meta[df_meta['label_norm'] == 'D'].copy()

    samples_a = df_a.sample(n=2, random_state=42).to_dict('records')
    samples_d = df_d.sample(n=2, random_state=42).to_dict('records')

    selected_samples = samples_a + samples_d
    random.shuffle(selected_samples)  # Random shuffle to avoid sequence hints

    ground_truth_dict = {}

    print(f"Generating 4 Blind Charts (FFT + STFT) in {OUTPUT_DIR}...\n")

    plt.style.use('dark_background')

    for idx, sample in enumerate(selected_samples, 1):
        blind_id = f"BLIND_SAMPLE_{idx:02d}"
        s_id = sample['sample_id']
        true_label = sample['label_norm']
        gt_status = "OK" if true_label == 'A' else "NG"

        s_path = sample['dest_path'] if 'dest_path' in sample and os.path.exists(sample['dest_path']) else os.path.join(RAW_ROOT, sample['machine_type'], s_id)
        sig = load_z8000_signal(s_path)

        if sig is None or len(sig) == 0:
            print(f"Warning: Could not load signal for {s_id}")
            continue

        ground_truth_dict[blind_id] = {
            "sample_id": s_id,
            "machine_type": sample.get('machine_type', 'Unknown'),
            "true_label": true_label,
            "ground_truth": gt_status
        }

        # --- Compute FFT ---
        N = len(sig)
        freqs = np.fft.rfftfreq(N, d=1.0/FS)
        fft_mag = np.abs(np.fft.rfft(sig)) / N
        fft_db = 20.0 * np.log10(fft_mag + 1e-6)

        # Limit to 0~5000 Hz
        fft_mask = freqs <= 5000.0
        freqs_sub = freqs[fft_mask]
        fft_db_sub = fft_db[fft_mask]

        # --- Compute STFT Spectrogram ---
        f_stft, t_stft, Zxx = stft(sig, fs=FS, nperseg=2048, noverlap=1536)
        stft_db = 20.0 * np.log10(np.abs(Zxx) + 1e-6)
        stft_mask = f_stft <= 5000.0
        f_stft_sub = f_stft[stft_mask]
        stft_db_sub = stft_db[stft_mask, :]

        # --- Plot Dual Panel Chart ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=300)

        # [Strict Blind Rule]: Title contains NO label or OK/NG keywords
        chart_title = f"Spindle Vibration Diagnosis Chart [ Blind Sample ID: #{idx:02d} ]"
        fig.suptitle(chart_title, fontsize=14, fontweight='bold', color='#E0E0E0', y=0.98)

        # 1) Top Subplot: FFT Power Spectrum (dB)
        ax1.plot(freqs_sub, fft_db_sub, color='#00E5FF', linewidth=0.8)
        ax1.set_xlim(0, 5000)
        ax1.set_ylabel("Power Spectrum (dB)", fontsize=11, fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.4, color='#555555')

        # Green dashed vertical lines for 1X, 2X (No text label)
        ax1.axvline(x=FR_1X, color='#00FF66', linestyle='--', linewidth=1.2, alpha=0.8)
        ax1.axvline(x=FR_2X, color='#00FF66', linestyle='--', linewidth=1.2, alpha=0.8)

        # Thick red dashed vertical lines for BPFO, BPFI (No text label)
        ax1.axvline(x=BPFO_FREQ, color='#FF3366', linestyle='--', linewidth=1.8, alpha=0.9)
        ax1.axvline(x=BPFI_FREQ, color='#FF3366', linestyle='--', linewidth=1.8, alpha=0.9)

        # 2) Bottom Subplot: STFT Time-Frequency Spectrogram
        mesh = ax2.pcolormesh(t_stft, f_stft_sub, stft_db_sub, cmap='inferno', shading='gouraud')
        ax2.set_xlim(0, t_stft[-1])
        ax2.set_ylim(0, 5000)
        ax2.set_xlabel("Time (s)", fontsize=11, fontweight='bold')
        ax2.set_ylabel("Frequency (Hz)", fontsize=11, fontweight='bold')
        cbar = fig.colorbar(mesh, ax=ax2, orientation='vertical', pad=0.02)
        cbar.set_label("Magnitude (dB)", fontsize=10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_img_path = os.path.join(OUTPUT_DIR, f"{blind_id}.png")
        plt.savefig(out_img_path, dpi=300)
        plt.close()

        print(f"  [+] Saved: {out_img_path}")

    # --- 4. Save Secret Ground Truth JSON ---
    json_path = os.path.join(OUTPUT_DIR, "blind_ground_truth.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth_dict, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(" Secret Ground Truth Mapping (Console Only)")
    print("=" * 60)
    print(json.dumps(ground_truth_dict, indent=4, ensure_ascii=False))
    print(f"\n[SUCCESS] Blind test charts & ground truth saved in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == '__main__':
    main()
