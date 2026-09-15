# PINO (Physics-Informed Neural Operator) 모델 평가 결과 및 개선 보고서

## 1. 개요 및 실험 구동 환경
* **평가 대상 파일**: [`evaluate_pino.py`](file:///c:/Users/user/Desktop/%EC%A0%9C%EC%A1%B0%20ai%20%EB%AA%A8%EB%8D%B8/evaluate_pino.py)
* **데이터셋 구성**: 총 244개 스핀들 샘플 (샘플당 8개 RPM 회전속도: 1000~8000 RPM, 3축 진동 신호)
* **검증 방법**: 5-Fold StratifiedGroupKFold Cross-Validation (동일 sample_id 데이터 누수 방지 및 샘플 단위 8개 RPM Soft Voting 결합)
* **기반 아키텍처**: 1D-FNO (Fourier Neural Operator) + RPM Conditioning + 가우시안 조화 진동 물리 손실(Harmonic Physics Loss)

---

## 2. 베이스라인 평가 결과 (5-Fold CV Soft Voting)

### 주요 성능 지표
- **전체 샘플 정확도 (Accuracy)**: **40.57%** (99 / 244)
- **Macro F1-Score**: **0.1631**
- **Weighted F1-Score**: **0.2647**

### 클래스별 상세 분류 성능 (Classification Report)
| 클래스 | Precision | Recall | F1-Score | Support (샘플 수) |
| :--- | :--- | :--- | :--- | :--- |
| **A (정상/양호)** | 0.50 | 0.04 | 0.08 | 92 |
| **B (경미/일반)** | 0.41 | **0.97** | **0.57** | 98 |
| **C (심각-BPFO)** | 0.00 | 0.00 | 0.00 | 11 |
| **D (치명-BPFI)** | 0.00 | 0.00 | 0.00 | 43 |
| **Total / Avg** | **0.35** | **0.41** | **0.16** | **244** |

### 혼동 행렬 (Confusion Matrix)
| 실제 \ 예측 | 예측 A | 예측 B | 예측 C | 예측 D |
| :--- | :---: | :---: | :---: | :---: |
| **실제 A** | **4** | **87** | 0 | 1 |
| **실제 B** | 2 | **95** | 1 | 0 |
| **실제 C** | 1 | **10** | **0** | 0 |
| **실제 D** | 1 | **42** | 0 | **0** |

---

## 3. 원인 분석 (Root Cause Analysis)

1. **다수 클래스(Class B) 쏠림 현상**:
   * 학습 초기 단계에서 모델이 가장 비율이 높은 B 클래스(98개)로 편향되어 244개 중 234개를 B로 예측함.
2. **에포크(Epoch) 부족**:
   * FNO 스펙트럼 도메인 파라미터와 가우시안 물리 손실(Harmonic Loss)이 수렴하기에는 10 Epoch 학습량이 부족함.
3. **소수 클래스(C: 11개, D: 43개) 가중치 부족**:
   * C, D 클래스의 샘플 수가 적어 CrossEntropy Loss 가중치 및 확률 thresholding 조정이 필요함.

---

## 4. 정확도 향상을 위한 향후 수정 계획 (Roadmap for `evaluate_pino.py`)

1. **학습 Epoch 증대 & Scheduler 적용**:
   * `Epoch=30~50`, `CosineAnnealingLR` 적용하여 깊은 스펙트럼 피처 학습
2. **Loss Weighting & Focal Loss**:
   * 클래스 역수 비중(`weight=[1.0, 0.94, 8.36, 2.13]`) 및 Focal Loss 도입으로 소수 클래스 감도 향상
3. **모델 용량(Capacity) 확장**:
   * `PINO_BearingClassifier(modes=128, width=128)` 설정으로 복잡한 주파수 도메인 분해 능력 상향
4. **Soft Voting Threshold / Temperature Scaling**:
   * 예측 logits 온도 스케일링으로 클래스 B 편향 보정
