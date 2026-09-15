import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from scipy.optimize import minimize

# ----------------------------------------------------
# [Step 1] 경로 및 실험 파라미터 정의
# ----------------------------------------------------
DATA_ROOT = "./data/raw"
META_CSV = "./data/meta/dataset_metadata.csv"
OUTPUT_DIR = "./data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_RPMS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
AXES = ['X', 'Y', 'Z']
FS = 25600  # Sampling Rate 25.6 kHz

# ----------------------------------------------------
# [Step 2] 고속 신호 로더 및 촘촘한 집계 피처 추출
# ----------------------------------------------------
def fast_read_signal(file_path, max_len=153600):
    if not os.path.exists(file_path):
        return np.array([], dtype=np.float32)
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [f.readline() for _ in range(max_len)]
            vals = [float(x.strip()) for x in lines if x.strip()]
            return np.array(vals, dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)

def extract_physics_metrics(sig, rpm, fs=25600):
    if len(sig) == 0:
        return None
    
    rms = float(np.sqrt(np.mean(sig**2)))
    peak = float(np.max(np.abs(sig)))
    mean = np.mean(sig)
    std = np.std(sig) + 1e-8
    kurt = float(np.mean(((sig - mean) / std)**4))
    
    fr = rpm / 60.0  # 1X
    n = len(sig)
    fft_vals = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    total_energy = float(np.sum(fft_vals**2)) + 1e-8
    
    # 1X (Class A)
    idx_1x = np.where((freqs >= fr - 2.0) & (freqs <= fr + 2.0))[0]
    ratio_1x = float(np.sum(fft_vals[idx_1x]**2)) / total_energy if len(idx_1x) > 0 else 0.0
    
    # 2X (Class B)
    idx_2x = np.where((freqs >= 2.0 * fr - 2.0) & (freqs <= 2.0 * fr + 2.0))[0]
    ratio_2x = float(np.sum(fft_vals[idx_2x]**2)) / total_energy if len(idx_2x) > 0 else 0.0
    
    # BPFO (Class C 11.334X)
    f_bpfo = 11.334 * fr
    idx_bpfo = np.where((freqs >= f_bpfo - 3.0) & (freqs <= f_bpfo + 3.0))[0]
    ratio_bpfo = float(np.sum(fft_vals[idx_bpfo]**2)) / total_energy if len(idx_bpfo) > 0 else 0.0

    # BPFI (Class D 13.666X)
    f_bpfi = 13.666 * fr
    idx_bpfi = np.where((freqs >= f_bpfi - 3.0) & (freqs <= f_bpfi + 3.0))[0]
    ratio_bpfi = float(np.sum(fft_vals[idx_bpfi]**2)) / total_energy if len(idx_bpfi) > 0 else 0.0

    return [rms, peak, kurt, ratio_1x, ratio_2x, ratio_bpfo, ratio_bpfi]

# ----------------------------------------------------
# [Step 3] 고밀도 집계 피처 데이터셋 생성
# ----------------------------------------------------
parquet_path = os.path.join(OUTPUT_DIR, "tabular_dataset_aggregated.parquet")

if not os.path.exists(parquet_path):
    print("\n[진행] RPM 고밀도 통합 집계 물리 피처 테이블 추출 중...")
    df_meta = pd.read_csv(META_CSV)
    df_meta = df_meta[df_meta['label'].notna()].reset_index(drop=True)
    
    rows = []
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="고밀도 피처 추출"):
        sample_path = row['dest_path']
        feat_dict = {
            'sample_id': row['sample_id'],
            'machine_type': row['machine_type'],
            'label': str(row['label']).strip()
        }
        
        metrics_all = []
        for rpm in TARGET_RPMS:
            for axis in AXES:
                fpath = os.path.join(sample_path, f"Raw_{axis}_{rpm}")
                sig = fast_read_signal(fpath)
                m = extract_physics_metrics(sig, rpm, FS)
                if m is not None:
                    metrics_all.append(m)
                    
        if len(metrics_all) > 0:
            arr_m = np.array(metrics_all) # [N_valid, 7]
            feat_dict['RMS_mean'] = float(np.mean(arr_m[:, 0]))
            feat_dict['RMS_max'] = float(np.max(arr_m[:, 0]))
            feat_dict['Peak_max'] = float(np.max(arr_m[:, 1]))
            feat_dict['Kurt_mean'] = float(np.mean(arr_m[:, 2]))
            feat_dict['Kurt_max'] = float(np.max(arr_m[:, 2]))
            feat_dict['Ratio_1X_mean'] = float(np.mean(arr_m[:, 3]))
            feat_dict['Ratio_1X_max'] = float(np.max(arr_m[:, 3]))
            feat_dict['Ratio_2X_mean'] = float(np.mean(arr_m[:, 4]))
            feat_dict['Ratio_2X_max'] = float(np.max(arr_m[:, 4]))
            feat_dict['BPFO_Ratio_mean'] = float(np.mean(arr_m[:, 5]))
            feat_dict['BPFO_Ratio_max'] = float(np.max(arr_m[:, 5]))
            feat_dict['BPFI_Ratio_mean'] = float(np.mean(arr_m[:, 6]))
            feat_dict['BPFI_Ratio_max'] = float(np.max(arr_m[:, 6]))
        else:
            for col in ['RMS_mean', 'RMS_max', 'Peak_max', 'Kurt_mean', 'Kurt_max', 
                        'Ratio_1X_mean', 'Ratio_1X_max', 'Ratio_2X_mean', 'Ratio_2X_max',
                        'BPFO_Ratio_mean', 'BPFO_Ratio_max', 'BPFI_Ratio_mean', 'BPFI_Ratio_max']:
                feat_dict[col] = 0.0
                
        rows.append(feat_dict)
        
    df_data = pd.DataFrame(rows)
    df_data.to_parquet(parquet_path, index=False)
    print(f"[완료] 고밀도 집계 데이터셋 저장 완료: {parquet_path}")
else:
    df_data = pd.read_parquet(parquet_path)

# ----------------------------------------------------
# [Step 4] 고밀도 피처 + 가중치/임계값 교차 검증
# ----------------------------------------------------
print("\n" + "="*50)
print("   고밀도 도메인 집계 피처 기반 머신러닝 성능 평가 (5-Fold CV)   ")
print("="*50)

df_features = pd.get_dummies(df_data.drop(columns=['sample_id', 'label']), columns=['machine_type'])
X = df_features.fillna(0.0).values

le = LabelEncoder()
y = le.fit_transform(df_data['label'])
classes = le.classes_

print(f"데이터 크기: {X.shape[0]} 샘플, {X.shape[1]} 개 고밀도 피처")
print(f"클래스 매핑: {dict(zip(range(len(classes)), classes))}")
print(f"클래스별 샘플 분포: {dict(pd.Series(y).value_counts())}\n")

sample_weights = compute_sample_weight(class_weight='balanced', y=y)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

try:
    from xgboost import XGBClassifier
    has_xgb = True
except ImportError:
    has_xgb = False

try:
    from lightgbm import LGBMClassifier
    has_lgb = True
except ImportError:
    has_lgb = False

def optimize_class_weights(y_true, probs):
    def loss_func(w):
        scaled_probs = probs * w
        preds = np.argmax(scaled_probs, axis=1)
        return -f1_score(y_true, preds, average='macro')
    
    init_w = np.ones(probs.shape[1])
    res = minimize(loss_func, init_w, method='Nelder-Mead', options={'maxiter': 1000})
    best_w = res.x
    best_preds = np.argmax(probs * best_w, axis=1)
    best_f1 = f1_score(y_true, best_preds, average='macro')
    best_acc = accuracy_score(y_true, best_preds)
    return best_w, best_preds, best_acc, best_f1

def run_cv(model_name, get_model, use_sample_weight=True):
    oof_probs = np.zeros((len(y), len(classes)))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        w_train = sample_weights[train_idx]
        
        clf = get_model()
        if use_sample_weight:
            clf.fit(X_train, y_train, sample_weight=w_train)
        else:
            clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_val)
        oof_probs[val_idx] = probs

    best_w, calib_preds, calib_acc, calib_f1 = optimize_class_weights(y, oof_probs)

    print("\n" + "="*50)
    print(f"   [{model_name}] 5-Fold Result")
    print("="*50)
    print(f"★ 임계값 최적화 -> Accuracy: {calib_acc*100:.2f}% | Macro F1: {calib_f1:.4f} ★")
    print("\n[상세 분류 보고서]")
    print(classification_report(y, calib_preds, target_names=[str(c) for c in classes]))
    print("[혼동 행렬]")
    cm = confusion_matrix(y, calib_preds)
    print(pd.DataFrame(cm, index=[f"실제_{c}" for c in classes], columns=[f"예측_{c}" for c in classes]))

    return oof_probs

# 1. Random Forest
get_rf = lambda: RandomForestClassifier(n_estimators=300, max_depth=6, class_weight='balanced', random_state=42)
rf_probs = run_cv("Random Forest (Class Weight Balanced)", get_rf, use_sample_weight=False)

# 2. Extra Trees
get_et = lambda: ExtraTreesClassifier(n_estimators=300, max_depth=6, class_weight='balanced', random_state=42)
et_probs = run_cv("Extra Trees (Class Weight Balanced)", get_et, use_sample_weight=False)

# 3. XGBoost
if has_xgb:
    get_xgb = lambda: XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.03, random_state=42, eval_metric='mlogloss')
    xgb_probs = run_cv("XGBoost (Sample Weight Balanced)", get_xgb, use_sample_weight=True)

# 4. LightGBM
if has_lgb:
    get_lgb = lambda: LGBMClassifier(n_estimators=250, max_depth=3, learning_rate=0.03, class_weight='balanced', random_state=42, verbose=-1)
    lgb_probs = run_cv("LightGBM (Class Weight Balanced)", get_lgb, use_sample_weight=False)

# 5. Ensemble
all_probs = [rf_probs, et_probs]
if has_xgb: all_probs.append(xgb_probs)
if has_lgb: all_probs.append(lgb_probs)

ens_probs = np.mean(all_probs, axis=0)
best_w_ens, calib_preds_ens, calib_acc_ens, calib_f1_ens = optimize_class_weights(y, ens_probs)

print("\n" + "="*50)
print("   ★ [최종 고밀도 피처 앙상블] 5-Fold Result ★")
print("="*50)
print(f"최종 Accuracy: {calib_acc_ens*100:.2f}% | Macro F1: {calib_f1_ens:.4f}")
print("\n[최종 앙상블 상세 분류 보고서]")
print(classification_report(y, calib_preds_ens, target_names=[str(c) for c in classes]))
print("[최종 앙상블 혼동 행렬]")
cm_ens = confusion_matrix(y, calib_preds_ens)
print(pd.DataFrame(cm_ens, index=[f"실제_{c}" for c in classes], columns=[f"예측_{c}" for c in classes]))
