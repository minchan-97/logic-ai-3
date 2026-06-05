[README.md](https://github.com/user-attachments/files/28627911/README.md)
# Topological Steering Streamlit API Demo

## 실행 방법

```bash
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

## API 모드

### 1. Mock
API 없이 데모 응답을 생성합니다.

### 2. OpenAI-compatible
OpenAI 호환 `/v1/chat/completions` 형식 API를 호출합니다.

필요한 환경변수:

```env
API_URL=https://api.openai.com/v1/chat/completions
API_KEY=your_api_key_here
MODEL_NAME=gpt-4o-mini
```

### 3. Custom JSON API
다음 형식으로 POST 요청을 보냅니다.

```json
{
  "prompt": "사용자 입력",
  "steering_mode": "사실 기반 정답 구멍 (Fact Hole)",
  "model": "모델명"
}
```

응답 JSON에서 아래 키 중 하나를 자동으로 읽습니다.

```json
{
  "answer": "응답"
}
```

지원 키: `answer`, `output`, `text`, `response`

## 주의

이 앱은 실제 LLM 내부 활성값을 읽는 시스템이 아니라, API 응답과 위상 궤도 제어 시뮬레이션을 결합한 PoC입니다.
