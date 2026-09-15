import os
import re
import glob
import pandas as pd
from tqdm import tqdm

RAW_ROOT = "./data/raw"
OUTPUT_DIR = "./data/meta"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_RPMS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
AXES = ['X', 'Y', 'Z']

def clean_meta_file(folder_path):
    info = {'SampleRate': 25600, 'target_label': None, 'qc_fft': None, 'qc_all': None}
    cand = glob.glob(os.path.join(folder_path, "Meta*"))
    if not cand:
        return info
        
    lines = []
    for enc in ['cp949', 'euc-kr', 'utf-8']:
        try:
            with open(cand[0], 'r', encoding=enc) as f:
                lines = f.readlines()
            if lines:
                break
        except Exception:
            continue
            
    for line in lines:
        clean = line.strip().replace(" ", "")
        if '관능평가:' in clean:
            val = clean.split('관능평가:')[-1].strip()
            # A, B, C, D 단일 등급 추출 (A+ -> A 등 표준화)
            if val:
                match = re.search(r"[ABCD]", val.upper())
                info['target_label'] = match.group(0) if match else val
        elif 'SampleRate:' in clean:
            val = clean.split('SampleRate:')[-1].strip()
            if val.isdigit():
                info['SampleRate'] = int(val)
        elif 'FFT평가:' in clean:
            info['qc_fft'] = clean.split('FFT평가:')[-1].strip() or None
        elif '종합평가:' in clean:
            info['qc_all'] = clean.split('종합평가:')[-1].strip() or None
    return info

records = []
machine_dirs = [d for d in os.listdir(RAW_ROOT) if os.path.isdir(os.path.join(RAW_ROOT, d))]

for m_dir in machine_dirs:
    m_path = os.path.join(RAW_ROOT, m_dir)
    sample_folders = [d for d in os.listdir(m_path) if os.path.isdir(os.path.join(m_path, d))]
    
    for s_folder in sample_folders:
        s_path = os.path.join(m_path, s_folder)
        meta = clean_meta_file(s_path)
        
        # 1000~8000 RPM X, Y, Z 24개 파일 존재 확인
        missing_count = 0
        for rpm in TARGET_RPMS:
            for axis in AXES:
                fname = f"Raw_{axis}_{rpm}"
                f_cand = glob.glob(os.path.join(s_path, f"{fname}*"))
                if not f_cand:
                    missing_count += 1
                    
        records.append({
            'sample_id': s_folder,
            'machine_type': m_dir,
            'target_label': meta['target_label'],  # A, B, C, D 정규화
            'qc_all': meta['qc_all'],              # OK, NG
            'qc_fft': meta['qc_fft'],              # 주파수평가 (결측치 허용)
            'sample_rate': meta['SampleRate'],
            'dest_path': s_path,                   # DataLoader 참조 경로
            'is_valid_24': (missing_count == 0),   # 24개 파일 온전성
            'missing_files': missing_count
        })

df_clean = pd.DataFrame(records)

# 엑셀 및 CSV 저장 (파일 열림 시 대비 처리)
excel_out = os.path.join(OUTPUT_DIR, "dataset_metadata_v2.xlsx")
csv_out = os.path.join(OUTPUT_DIR, "dataset_metadata.csv")

try:
    df_clean.to_excel(excel_out, index=False)
    print(f"엑셀 저장 완료: {excel_out}")
except Exception as e:
    print(f"엑셀 저장 경고: {e}")

try:
    df_clean.to_csv(csv_out, index=False, encoding='utf-8-sig')
    print(f"CSV 저장 완료: {csv_out}")
except PermissionError:
    alt_csv = os.path.join(OUTPUT_DIR, "dataset_metadata_v2.csv")
    df_clean.to_csv(alt_csv, index=False, encoding='utf-8-sig')
    print(f"경고: {csv_out} 파일이 열려 있어서 {alt_csv} 로 대체 저장되었습니다.")
except Exception as e:
    print(f"CSV 저장 오류: {e}")

print(f"\n정리 완료: 총 {len(df_clean)}개 샘플")
print(df_clean[['machine_type', 'target_label', 'is_valid_24']].value_counts())

