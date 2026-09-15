import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 설정 및 상치 정의
DATA_META_PATH = "./data/meta/dataset_metadata.csv"
OUTPUT_DIR = "./poc_charts"
FS = 25600  # 샘플링 주파수 (25.6 kHz)
TARGET_RPM = 8000  # 회전 속도 (8000 RPM)
TARGET_AXIS = "Z"  # 고속 회전 시 충격이 뚜렷한 Z축

# 베어링 기구학 제원 (RPM=8000 기준, fr = 8000 / 60 ≈ 133.33 Hz)
FR = TARGET_RPM / 60.0
FREQ_1X = FR          # 약 133.33 Hz
FREQ_2X = 2.0 * FR    # 약 266.67 Hz
FREQ_BPFO = 11.334 * FR  # 약 1,511.20 Hz (외륜 결함)
FREQ_BPFI = 13.666 * FR  # 약 1,822.13 Hz (내륜 결함)

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_signal(sample_path, rpm=8000, axis="Z", max_points=153600):
    """지정 폴더에서 Raw_[AXIS]_[RPM].txt 신호 로드 (최대 6초: 153,600개)"""
    file_name = f"Raw_{axis}_{rpm}.txt"
    file_path = os.path.join(sample_path, file_name)
    if not os.path.exists(file_path):
        file_path = os.path.join(sample_path, f"Raw_{axis}_{rpm}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"신호 파일을 찾을 수 없습니다: {file_path}")

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        vals = [float(line.strip()) for line in f if line.strip()]
    
    sig = np.array(vals[:max_points], dtype=np.float32)
    return sig

def plot_vlm_poc_chart(sample_id, label, sig, save_path):
    """VLM 판독용 고해상도 2D 진단 차트 생성 (FFT + STFT 2-Subplot)"""
    fig, (ax_fft, ax_stft) = plt.subplots(2, 1, figsize=(14, 10), dpi=300)
    
    # ----------------------------------------------------
    # [1] 상단: 순수 FFT 파워 스펙트럼 (dB 스케일)
    # ----------------------------------------------------
    n_pts = len(sig)
    fft_vals = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(n_pts, d=1.0 / FS)
    
    # dB 스케일 변환 (20 * log10(amplitude + 1e-6))
    fft_db = 20.0 * np.log10(fft_vals + 1e-6)
    
    # 0 ~ 5,000 Hz 범위 마스킹
    freq_mask = freqs <= 5000.0
    freqs_cut = freqs[freq_mask]
    fft_db_cut = fft_db[freq_mask]
    
    ax_fft.plot(freqs_cut, fft_db_cut, color='#1f77b4', linewidth=0.8, label='FFT Power Spectrum (dB)')
    
    # 물리 가이드라인 오버레이
    # 1X, 2X 초록색 수직 점선
    ax_fft.axvline(x=FREQ_1X, color='green', linestyle='--', linewidth=1.2, alpha=0.8, label='1X (133.3 Hz)')
    ax_fft.axvline(x=FREQ_2X, color='darkgreen', linestyle='--', linewidth=1.2, alpha=0.8, label='2X (266.7 Hz)')
    
    # BPFO, BPFI 굵은 빨간색 수직 점선 및 마킹
    ax_fft.axvline(x=FREQ_BPFO, color='red', linestyle='--', linewidth=1.8, alpha=0.9, label='BPFO (1511.2 Hz)')
    ax_fft.axvline(x=FREQ_BPFI, color='crimson', linestyle='--', linewidth=1.8, alpha=0.9, label='BPFI (1822.1 Hz)')
    
    # 텍스트 마킹 추가
    max_db = np.max(fft_db_cut)
    ax_fft.text(FREQ_1X, max_db - 5, ' 1X', color='green', fontsize=9, fontweight='bold')
    ax_fft.text(FREQ_2X, max_db - 5, ' 2X', color='darkgreen', fontsize=9, fontweight='bold')
    ax_fft.text(FREQ_BPFO, max_db - 5, ' BPFO', color='red', fontsize=10, fontweight='bold')
    ax_fft.text(FREQ_BPFI, max_db - 12, ' BPFI', color='crimson', fontsize=10, fontweight='bold')
    
    ax_fft.set_xlim(0, 5000)
    ax_fft.set_title(f"Sample: {sample_id} | True Label: {label} | 8000 RPM Z-Axis FFT Spectrum", fontsize=12, fontweight='bold')
    ax_fft.set_xlabel("Frequency (Hz)", fontsize=10)
    ax_fft.set_ylabel("Amplitude (dB)", fontsize=10)
    ax_fft.grid(True, which='both', linestyle=':', alpha=0.5)
    ax_fft.legend(loc='upper right', fontsize=8)
    
    # ----------------------------------------------------
    # [2] 하단: STFT 스펙트로그램 (2D 히트맵)
    # ----------------------------------------------------
    f_stft, t_stft, Zxx = signal.stft(sig, fs=FS, nperseg=1024, noverlap=512)
    Zxx_db = 20.0 * np.log10(np.abs(Zxx) + 1e-6)
    
    stft_mask = f_stft <= 5000.0
    f_stft_cut = f_stft[stft_mask]
    Zxx_db_cut = Zxx_db[stft_mask, :]
    
    mesh = ax_stft.pcolormesh(t_stft, f_stft_cut, Zxx_db_cut, cmap='inferno', shading='gouraud')
    fig.colorbar(mesh, ax=ax_stft, label='Power (dB)')
    
    ax_stft.set_ylim(0, 5000)
    ax_stft.set_title(f"Short-Time Fourier Transform (STFT) Spectrogram (0~5000 Hz)", fontsize=11, fontweight='bold')
    ax_stft.set_xlabel("Time (s)", fontsize=10)
    ax_stft.set_ylabel("Frequency (Hz)", fontsize=10)
    
    plt.suptitle(f"VLM PoC Diagnosis Chart: {sample_id} [Label: {label}]", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def main():
    if not os.path.exists(DATA_META_PATH):
        raise FileNotFoundError(f"메타데이터 CSV가 없습니다: {DATA_META_PATH}")
        
    df_meta = pd.read_csv(DATA_META_PATH)
    
    # [1] 샘플 선별 로직
    # 양품 (OK): label == 'A' 2개
    ok_df = df_meta[df_meta['label'] == 'A'].head(2)
    
    # 불량 (NG): label == 'D' (치명 내륜 결함) 2개 (부족 시 C 또는 qc_all NG 포함)
    ng_df = df_meta[df_meta['label'] == 'D'].head(2)
    if len(ng_df) < 2:
        needed = 2 - len(ng_df)
        c_df = df_meta[df_meta['label'] == 'C'].head(needed)
        ng_df = pd.concat([ng_df, c_df])
        
    selected_ok = ok_df.to_dict('records')
    selected_ng = ng_df.to_dict('records')
    
    print("\n" + "="*70)
    print("   [VLM PoC 시각화] OK & NG 대표 샘플 4종 차트 자동 생성 시작   ")
    print("="*70)
    
    generated_files = []
    
    # OK 샘플 생성
    for idx, item in enumerate(selected_ok, 1):
        s_id = item['sample_id']
        lbl = item['label']
        path = item['dest_path']
        sig = load_signal(path, rpm=TARGET_RPM, axis=TARGET_AXIS)
        save_file = os.path.join(OUTPUT_DIR, f"POC_OK_sample{idx}_{s_id}.png")
        plot_vlm_poc_chart(s_id, lbl, sig, save_file)
        generated_files.append((f"OK Sample {idx}", s_id, lbl, save_file))
        print(f"[OK 생성 완료] Sample: {s_id} (Label: {lbl}) -> {save_file}")
        
    # NG 샘플 생성
    for idx, item in enumerate(selected_ng, 1):
        s_id = item['sample_id']
        lbl = item['label']
        path = item['dest_path']
        sig = load_signal(path, rpm=TARGET_RPM, axis=TARGET_AXIS)
        save_file = os.path.join(OUTPUT_DIR, f"POC_NG_sample{idx}_{s_id}.png")
        plot_vlm_poc_chart(s_id, lbl, sig, save_file)
        generated_files.append((f"NG Sample {idx}", s_id, lbl, save_file))
        print(f"[NG 생성 완료] Sample: {s_id} (Label: {lbl}) -> {save_file}")

    print("\n" + "="*70)
    print("   [PoC 시각화 차트 생성 완료 및 VLM 테스트 안내]   ")
    print("="*70)
    print("생성된 고해상도(300 DPI) 진단 차트 4장 위치:")
    for category, s_id, lbl, save_path in generated_files:
        print(f"  - {category}: {s_id} (Label: {lbl}) -> {save_path}")
        
    print("\n[OpenRouter VLM 테스트 가이드]")
    print("1. .env 파일에 OPENROUTER_API_KEY 및 OPENROUTER_MODEL (예: openai/gpt-4o) 설정 확인")
    print("2. 생성된 PNG 차트 이미지를 OpenRouter 멀티모달 API에 전달하여 시각적 진단 성능 검증 수행 가능")
    print("3. 자세한 실행 및 프롬프트 가이드는 TEST.MD 파일에 기록되었습니다.")

if __name__ == "__main__":
    main()
