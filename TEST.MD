# VLM (Vision-Language Model) 공작기계 베어링 진동 차트 판독 PoC 검증 가이드

## 1. 개요 및 검증 목적
수치 데이터 기반의 딥러닝(PINO) 모델에서 소수 클래스(불량 C, D등급) 탐지율이 저조함에 따라, 사전학습 VLM(GPT-4o, Claude 3.5 Sonnet 등)이 주파수 차트를 시각적으로 판독할 수 있는지 개념 검증(PoC)을 수행합니다.

`visualize_poc_samples.py`를 통해 양품(OK) 2개 및 불량(NG) 2개 샘플의 고해상도(300 DPI) 2D 진단 차트(FFT 스펙트럼 + STFT 스펙트로그램)를 자동 생성하였으며, **OpenRouter API**를 연동하여 시각적 결함 판독 실험을 수행합니다.

---

## 2. 환경 설정 (`.env`)
프로젝트 루트 디렉토리의 `.env` 파일에 OpenRouter API 설정값을 지정합니다.

```env
# OpenRouter API Key & Configuration for VLM PoC Evaluation
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openai/gpt-4o
# OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

---

## 3. PoC 시각화 차트 생성 (`visualize_poc_samples.py`)

### 차트 구성 특징
1. **[상단] 순수 FFT 파워 스펙트럼 (0~5,000 Hz)**:
   - Amplitude를 dB 스케일(`20 * log10(FFT_amplitude + 1e-6)`)로 변환
   - **1X (133.3 Hz)**, **2X (266.7 Hz)**: 초록색 수직 점선 오버레이
   - **BPFO (1,511.2 Hz)**, **BPFI (1,822.1 Hz)**: 굵은 빨간색 수직 점선 오버레이 및 마킹 텍스트 명시
2. **[하단] STFT 스펙트로그램 (2D 히트맵)**:
   - `scipy.signal.stft` 사용 (nperseg=1024, noverlap=512)
   - 주파수 0~5,000 Hz, 컬러맵 `'inferno'`, dB 정규화 에너지 분포

### 생성된 PoC 샘플 차트 목록 (`./poc_charts/`)
- **[OK 1]** `POC_OK_sample1_DH_KF5600C__20180611_1325.png` (Label: A - 정상)
- **[OK 2]** `POC_OK_sample2_DH_KF5600C__20180616_1058.png` (Label: A - 정상)
- **[NG 1]** `POC_NG_sample1_DH_KF5600C__20180619_1440.png` (Label: D - 치명 내륜 결함)
- **[NG 2]** `POC_NG_sample2_DH_KF5600C__20180723_1445.png` (Label: D - 치명 내륜 결함)

---

## 4. OpenRouter 연동 VLM 자동 판독 클라이언트 코드 (`vlm_eval_openrouter.py`)

생성된 차트 이미지를 OpenRouter API(GPT-4o / Claude 3.5 Sonnet)로 전송하여 자동 진단을 수행하는 파이썬 실행 코드입니다.

```python
import os
import base64
import json
import requests
from dotenv import load_dotenv

# .env 로드
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

def encode_image_to_base64(image_path):
    """이미지 파일을 base64 문자열로 인코딩"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def diagnose_chart_with_vlm(image_path):
    """OpenRouter API를 이용해 VLM 차트 진단 호출"""
    base64_image = encode_image_to_base64(image_path)
    
    prompt = """
[역할] 당신은 공작기계 스핀들 베어링 진동 신호 분석 최고 전문가입니다.
제시된 2D 진단 차트는 8,000 RPM 고속 회전 시 Z축 진동 신호의 (상단) FFT 파워 스펙트럼과 (하단) STFT 스펙트로그램입니다.

[물리적 제원 정보 (8000 RPM 기준)]
- 1X (회전 기본주파수): 133.3 Hz
- 2X (고조파): 266.7 Hz
- BPFO (외륜 결함): 1,511.2 Hz (빨간색 점선)
- BPFI (내륜 결함): 1,822.1 Hz (빨간색 점선)

[분석 지침]
1. 상단 FFT 차트에서 빨간색 점선으로 표시된 BPFO(1511Hz) 또는 BPFI(1822Hz) 주변에 유의미한 진폭 피크(dB)가 관측되는지 확인하세요.
2. 하단 STFT 스펙트로그램에서 1500~2000Hz 주파수 대역에 지속적인 충격성 에너지 밴드가 형성되어 있는지 분석하세요.
3. 차트 분석 결과를 바탕으로 최종 상태를 '양품 (OK)' 또는 '불량 (NG)'으로 판정하고 근거를 제시하세요.

[응답 양식 (JSON)]
{
  "diagnosis": "OK 또는 NG",
  "fault_type": "None / BPFO / BPFI / Unbalance",
  "confidence": 0.0~1.0,
  "reasoning": "상세 분석 소경 및 근거"
}
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://localhost",
        "X-Title": "Spindle Vibration VLM PoC",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "response_format": {"type": "json_object"} if "gpt-4" in OPENROUTER_MODEL else None
    }

    response = requests.post(f"{OPENROUTER_BASE_URL}/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        res_json = response.json()
        content = res_json["choices"][0]["message"]["content"]
        return content
    else:
        return f"Error {response.status_code}: {response.text}"

if __name__ == "__main__":
    poc_charts = [
        "./poc_charts/POC_OK_sample1_DH_KF5600C__20180611_1325.png",
        "./poc_charts/POC_OK_sample2_DH_KF5600C__20180616_1058.png",
        "./poc_charts/POC_NG_sample1_DH_KF5600C__20180619_1440.png",
        "./poc_charts/POC_NG_sample2_DH_KF5600C__20180723_1445.png"
    ]
    
    print("="*60)
    print("  [OpenRouter VLM 차트 진단 PoC 실행]  ")
    print("="*60)
    
    for chart in poc_charts:
        if os.path.exists(chart):
            print(f"\n▶ 진단 대상 차트: {chart}")
            result = diagnose_chart_with_vlm(chart)
            print("▶ VLM 진단 결과:")
            print(result)
```

---

## 5. 기대 효과 및 검증 지표
- **시각적 가이드라인 효과**: 1X/2X 및 BPFO/BPFI 물리 가이드라인 오버레이를 통해 Zero-Shot 상황에서도 VLM이 정확한 결함 주파수를 식별할 수 있습니다.
- **PINO 대치 및 하이브리드 검증**: 수치 기반 FNO 모델과 VLM 비전 판독을 결합한 하이브리드 진단 파이프라인 구축 가능성을 확인합니다.
