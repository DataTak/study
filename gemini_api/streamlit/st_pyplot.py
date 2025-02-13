# st_pyplot.py
import streamlit as st
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 설치된 폰트 
plt.rcParams['font.family'] = 'NanumGothic'


# 로짓값을 가진 단어 목록
vocab_logits = {
    "나는": 0.01, "내일": 0.03, "오늘": 0.25, "어제": 0.3,
    "산에": 0.4, "학교에": 0.5, "집에": 0.65,
    "오른다": 1.2, "간다": 1.05, "왔다": 0.95
}

# 소프트맥스 변환 함수 (안정성 개선)
def softmax_with_temperature(values, temperature):
    epsilon = 1e-2    
    temperature = max(temperature, epsilon)
    values = np.array(values)

    max_logit = np.max(values)
    exp_values = np.exp((values - max_logit) / temperature)
    sum_exp_values = np.sum(exp_values)

    return (exp_values / sum_exp_values).tolist()

# 바 그래프 그리기 함수
def draw_prob_graph(vocab, probs):
    fig = plt.figure(figsize=(8, 4))
    
    # 색상 매칭 오류 방지
    palette_as_list = sns.color_palette("Reds", n_colors=len(vocab))

    # 확률 기준 정렬 후 리스트로 변환
    sorted_vocab_prob = sorted(zip(vocab, probs), key=lambda x: x[1])
    sorted_vocab, sorted_probs = zip(*sorted_vocab_prob)

    sorted_vocab = list(sorted_vocab)  # 튜플을 리스트로 변환
    sorted_probs = list(sorted_probs)  # 튜플을 리스트로 변환

    print(f"sorted_vocab: {sorted_vocab}")  # 디버깅용
    print(f"sorted_probs: {sorted_probs}")  # 디버깅용

    # Seaborn barplot (hue 제거)
    sns.barplot(x=sorted_vocab, y=sorted_probs, palette=palette_as_list)
    plt.xticks(rotation=45)  # 가독성 향상
    plt.ylabel("확률값")
    plt.xlabel("단어")

    st.pyplot(fig)

# Streamlit 슬라이더 설정
temperature = st.slider("Temperature 값 조정", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

# 확률값 변환
vocab = list(vocab_logits.keys())
logits = list(vocab_logits.values())
probs = softmax_with_temperature(logits, temperature=temperature)

# 그래프 출력
draw_prob_graph(vocab, probs)
