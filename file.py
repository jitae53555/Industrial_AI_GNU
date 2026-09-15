import os
import re
import glob
import shutil
from tqdm import tqdm
import pandas as pd

RAW_SOURCES = [
    r"C:\Users\user\Desktop\3학년 2학기\품질빅데이터\drive-download-20260914T030707Z-1-003",
    r"C:\Users\user\Desktop\3학년 2학기\품질빅데이터\drive-download-20260914T030707Z-1-002",
    r"C:\Users\user\Desktop\3학년 2학기\품질빅데이터\drive-download-20260914T030707Z-1-001"
]

TARGET_ROOT = r"./data/raw"
META_OUT_DIR = r"./data/meta"
os.makedirs(TARGET_ROOT, exist_ok=True)
os.makedirs(META_OUT_DIR, exist_ok=True)

def parse_meta(folder_path):
    info = {'SampleRate': 25600, 'label': None, 'qc_fft': None, 'qc_all': None}
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
            info['label'] = clean.split('관능평가:')[-1].strip() or None
        elif 'SampleRate:' in clean:
            val = clean.split('SampleRate:')[-1].strip()
            if val.isdigit():
                info['SampleRate'] = int(val)
        elif 'FFT평가:' in clean:
            info['qc_fft'] = clean.split('FFT평가:')[-1].strip()
        elif '종합평가:' in clean:
            info['qc_all'] = clean.split('종합평가:')[-1].strip()
    return info

# 1단계: 유효 폴더 수집
all_folders = []
for p in RAW_SOURCES:
    if os.path.exists(p):
        for d in os.listdir(p):
            full_d = os.path.join(p, d)
            if os.path.isdir(full_d) and d.startswith("DH_"):
                all_folders.append(full_d)

print(f"\n[발견] 복사 대상 폴더: 총 {len(all_folders)}개\n")

meta_records = []

# 2단계: 복사 및 500 RPM 배제
for folder_path in tqdm(all_folders, desc="파일 분류 및 복사 진행"):
    folder_name = os.path.basename(folder_path)
    
    # 기종명 추출 로직 완화
    tokens = folder_name.split("__")
    machine_type = tokens[0]
    
    test_date, test_time = "Unknown", "Unknown"
    if len(tokens) > 1:
        date_time_match = re.search(r"(\d{8})_(\d{4})", tokens[1])
        if date_time_match:
            test_date, test_time = date_time_match.groups()

    dest_dir = os.path.join(TARGET_ROOT, machine_type, folder_name)
    os.makedirs(dest_dir, exist_ok=True)
    
    meta = parse_meta(folder_path)
    meta_records.append({
        'sample_id': folder_name,
        'machine_type': machine_type,
        'test_date': test_date,
        'test_time': test_time,
        'label': meta['label'],
        'qc_all': meta['qc_all'],
        'sample_rate': meta['SampleRate'],
        'dest_path': dest_dir
    })
    
    for file_name in os.listdir(folder_path):
        if "_500" in file_name:
            continue
        
        src_file = os.path.join(folder_path, file_name)
        dst_file = os.path.join(dest_dir, file_name)
        
        # 파일이 없거나 크기가 다를 경우 정상 복사
        if os.path.isfile(src_file):
            if not os.path.exists(dst_file) or os.path.getsize(src_file) != os.path.getsize(dst_file):
                shutil.copy2(src_file, dst_file)

# 3단계: 메타 저장
df_meta = pd.DataFrame(meta_records)
df_meta.to_csv(os.path.join(META_OUT_DIR, "dataset_metadata.csv"), index=False, encoding='utf-8-sig')

print(f"\n[완료] 총 정리 대상: {len(all_folders)}개 중 {len(df_meta)}개 메타 기록 완료")