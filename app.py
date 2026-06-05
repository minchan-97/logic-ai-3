import os
import time
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Topological Steering API Demo", layout="wide")


# =========================================================
# 1. Topological landscape generation
# =========================================================
@st.cache_resource
def generate_topological_landscape() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a nonlinear activation landscape with two attractor holes."""
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)

    Z = np.sin(X) * np.cos(Y) + 0.5 * np.sin(2 * X) * np.sin(2 * Y)
    Z -= 1.5 * np.exp(-((X - 1.0) ** 2 + (Y - 1.0) ** 2) / 0.3)      # Fact hole
    Z -= 2.0 * np.exp(-((X + 1.5) ** 2 + (Y + 1.5) ** 2) / 0.4)      # Hallucination hole
    return X, Y, Z


X, Y, Z = generate_topological_landscape()


# =========================================================
# 2. API call layer
# =========================================================
def call_llm_api(
    provider: str,
    prompt: str,
    steering_mode: str,
    api_url: str,
    api_key: str | None,
    model: str,
) -> str:
    """
    Calls either a local/custom API, OpenAI-compatible API, or returns a mock response.

    Supported provider values:
    - Mock
    - OpenAI-compatible
    - Custom JSON API
    """
    if provider == "Mock":
        if steering_mode == "사실 기반 정답 구멍 (Fact Hole)":
            return "대한민국의 수도는 서울이며, 행정·정치·경제의 중심지입니다."
        return "우주 비행선이 조선시대 한양에 불시착하여 설립되었습니다. (환각 예시)"

    if not api_url:
        raise ValueError("API URL이 비어 있습니다.")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if provider == "OpenAI-compatible":
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a controlled reasoning demo model. "
                        "Answer briefly and directly."
                    ),
                },
                {
                    "role": "user",
                    "content": f"[Steering target: {steering_mode}]\n{prompt}",
                },
            ],
            "temperature": 0.3,
        }
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    # Custom JSON API: expects JSON response with one of these keys:
    # answer, output, text, response
    payload = {
        "prompt": prompt,
        "steering_mode": steering_mode,
        "model": model,
    }
    response = requests.post(api_url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    for key in ["answer", "output", "text", "response"]:
        if key in data:
            return str(data[key])

    return str(data)


def split_for_streaming(text: str) -> List[str]:
    """Simple token-like streaming chunks."""
    chunks = text.replace("\n", " \n ").split(" ")
    return [chunk + " " for chunk in chunks if chunk != ""]


# =========================================================
# 3. Visualization helpers
# =========================================================
def nearest_z(x: float, y: float) -> float:
    xi = np.abs(X[0] - x).argmin()
    yi = np.abs(Y[:, 0] - y).argmin()
    return float(Z[yi, xi])


def build_figure(
    path_x: List[float],
    path_y: List[float],
    path_z: List[float],
    target_x: float,
    target_y: float,
    target_color: str,
) -> go.Figure:
    target_z = nearest_z(target_x, target_y) - 0.5

    fig = go.Figure(
        data=[
            go.Surface(z=Z, x=X, y=Y, colorscale="Viridis", opacity=0.7, showscale=False),
            go.Scatter3d(
                x=[target_x],
                y=[target_y],
                z=[target_z],
                mode="markers",
                marker=dict(size=12, color=target_color, symbol="diamond"),
                name="Target Hole",
            ),
            go.Scatter3d(
                x=path_x,
                y=path_y,
                z=path_z,
                mode="lines+markers",
                line=dict(color="white", width=6),
                marker=dict(size=5, color="orange"),
                name="Reasoning Trajectory",
            ),
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="X Dimension / Concept Space",
            yaxis_title="Y Dimension / Context Space",
            zaxis_title="Z Energy / Activation",
            aspectratio=dict(x=1, y=1, z=0.6),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig


# =========================================================
# 4. UI
# =========================================================
st.title("🛡️ Topological Steering API Demo")
st.markdown(
    """
이 앱은 LLM 응답을 API로 호출한 뒤, 응답 생성 과정을 **위상수학적 활성화 제어 시뮬레이션**으로 시각화하는 Streamlit PoC입니다.  
실제 LLM 내부 활성값을 읽는 앱은 아니며, API 응답 텍스트와 별도의 궤도 제어 시뮬레이션을 결합한 데모입니다.
"""
)
st.divider()

with st.sidebar:
    st.header("🎛️ Steering Control")

    steering_mode = st.radio(
        "1. 목표 결론 구멍(Target Hole)",
        ["사실 기반 정답 구멍 (Fact Hole)", "우회/환각 구멍 (Hallucination)"],
    )

    steering_strength = st.slider(
        "2. 강제 개입 세기 (Steering Strength c)",
        min_value=0.0,
        max_value=4.0,
        value=1.5,
        step=0.1,
    )

    noise_level = st.slider(
        "3. 연산 노이즈 수준",
        min_value=0.1,
        max_value=1.0,
        value=0.4,
        step=0.1,
    )

    st.divider()
    st.header("🔌 API Settings")

    provider = st.selectbox(
        "API Provider",
        ["Mock", "OpenAI-compatible", "Custom JSON API"],
    )

    default_url = os.getenv("API_URL", "")
    default_key = os.getenv("API_KEY", "")
    default_model = os.getenv("MODEL_NAME", "gpt-4o-mini")

    api_url = st.text_input(
        "API URL",
        value=default_url,
        placeholder="예: https://api.openai.com/v1/chat/completions 또는 http://localhost:3000/analyze",
    )
    api_key = st.text_input("API Key", value=default_key, type="password")
    model = st.text_input("Model", value=default_model)


if steering_mode == "사실 기반 정답 구멍 (Fact Hole)":
    target_x, target_y = 1.0, 1.0
    target_color = "green"
else:
    target_x, target_y = -1.5, -1.5
    target_color = "red"

st.subheader("🚀 Prompt & Inference")
user_prompt = st.text_input("테스트할 질문을 입력하세요", "대한민국의 수도는 어디인가요?")

run = st.button("API 호출 및 위상 궤도 제어 시작", type="primary")

if run:
    col1, col2 = st.columns([2, 1])

    with col1:
        chart_placeholder = st.empty()
    with col2:
        st.write("### 📝 API Output Stream")
        text_placeholder = st.empty()
        log_placeholder = st.empty()

    try:
        with st.spinner("API 호출 중..."):
            api_output = call_llm_api(
                provider=provider,
                prompt=user_prompt,
                steering_mode=steering_mode,
                api_url=api_url,
                api_key=api_key,
                model=model,
            )
    except Exception as exc:
        st.error(f"API 호출 실패: {exc}")
        st.stop()

    chunks = split_for_streaming(api_output)
    if not chunks:
        chunks = [api_output]

    current_x, current_y = -0.5, 0.5
    path_x = [current_x]
    path_y = [current_y]
    path_z = [nearest_z(current_x, current_y)]
    generated_text = ""

    for step, chunk in enumerate(chunks, start=1):
        time.sleep(0.25)

        dir_x = target_x - current_x
        dir_y = target_y - current_y
        noise_x = np.random.randn() * noise_level
        noise_y = np.random.randn() * noise_level

        current_x += noise_x + (steering_strength * 0.2 * dir_x)
        current_y += noise_y + (steering_strength * 0.2 * dir_y)

        current_x = float(np.clip(current_x, -4, 4))
        current_y = float(np.clip(current_y, -4, 4))
        current_z = nearest_z(current_x, current_y)

        path_x.append(current_x)
        path_y.append(current_y)
        path_z.append(current_z)

        generated_text += chunk
        text_placeholder.code(generated_text, language="markdown")
        log_placeholder.markdown(
            f"""
- **현재 단계:** {step} / {len(chunks)}
- **연산 공 좌표:** `({current_x:.2f}, {current_y:.2f})`
- **스티어링 강도:** `{steering_strength}`
- **노이즈 수준:** `{noise_level}`
"""
        )

        fig = build_figure(path_x, path_y, path_z, target_x, target_y, target_color)
        chart_placeholder.plotly_chart(fig, use_container_width=True)

    st.success(f"🎯 제어 완료: `{steering_mode}` 방향으로 응답 궤도 시뮬레이션을 종료했습니다.")

    with st.expander("Raw API Output"):
        st.write(api_output)
