import os
import base64
import requests
from dotenv import load_dotenv

# .env 로드
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

def encode_image_to_base64(image_path):
    """이미지 파일을 base64 문자열로 인코딩"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def diagnose_chart_with_vlm(image_path):
    """OpenRouter API를 이용해 VLM 차트 진단 호출"""
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        return "[경고] .env 파일에 올바른 OPENROUTER_API_KEY를 입력해 주세요."

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
  "confidence": 0.85,
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
        ]
    }

    try:
        response = requests.post(f"{OPENROUTER_BASE_URL}/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            res_json = response.json()
            content = res_json["choices"][0]["message"]["content"]
            return content
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Exception occurred: {e}"

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
