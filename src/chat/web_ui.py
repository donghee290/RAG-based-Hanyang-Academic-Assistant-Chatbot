import streamlit as st
import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 설정
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.chat.chat_server import ChatSession

def apply_custom_styles():
    st.markdown(f"""
    <style>
        :root {{
            color-scheme: light;
        }}

        @media (prefers-color-scheme: dark) {{
            html, body, [data-testid="stAppViewContainer"] {{
                background-color: #FFFFFF !important;
                color: #000000 !important;
            }}
        }}

        [data-testid="stHeader"],
        [data-testid="stToolbar"] {{
            display: none !important;
        }}

        /* 기본 */
        html, body {{
            font-size: 22px;
            font-family: 'Pretendard', 'Malgun Gothic', sans-serif;
            background-color: #FFFFFF;
            color: #000000;
        }}

        /* 메인 컨테이너(하단 입력창 떠있게 쓰는 경우) */
        .block-container {{
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 40vh;
        }}

        /* 헤더 */
        .main-header {{
            font-size: 3rem;
            color: #0E4A84;
            font-weight: 550;
            margin-bottom: 0.5rem;
        }}

        /* 채팅 메시지 박스 */
        div[data-testid="stChatMessage"] {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 20px;
            padding: 1.5rem;
            margin-bottom: 0.8rem;
            font-size: 1.2rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            color: #000000;
        }}
        div[data-testid="stChatMessage"] * {{
            color: #000000 !important;
        }}

        /* 아바타(여기가 핵심): Streamlit 버전별 DOM 차이를 감안해 “첫 번째 컬럼(아바타 영역)”까지 같이 강제 */
        div[data-testid="stChatMessage"] > div:first-child,
        div[data-testid="stChatMessageAvatar"],
        div[data-testid="stChatMessageAvatar"] * {{
            background-color: #FFFFFF !important;
            color: #000000 !important;
            border-radius: 9999px !important;
        }}

        /* 일부 버전에서 아바타가 svg/icon wrapper로 들어가는 케이스까지 */
        div[data-testid="stChatMessage"] svg,
        div[data-testid="stChatMessage"] img {{
            background-color: transparent !important;
        }}

        /* 입력창 고정 */
        div[data-testid="stChatInput"] {{
            position: fixed;
            left: 50%;
            transform: translateX(-50%);
            bottom: 30vh;
            width: min(900px, 100%);
            z-index: 1000;
            background: transparent;
        }}
        div[data-testid="stChatInput"] > div {{
            background-color: #FFFFFF;
            padding: 10px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.15);
        }}
        div[data-testid="stChatInput"] textarea,
        div[data-testid="stChatInput"] input {{
            background-color: #FFFFFF;
            color: #000000;
        }}

        div.stButton > button,
        div.stButton button,
        div[data-testid="stButton"] > button,
        div[data-testid="stButton"] button,
        button[kind],
        button[kind="secondary"],
        button[kind="primary"],
        [data-testid^="baseButton-"] {{
            background: #FFFFFF !important;
            background-color: #FFFFFF !important;
            color: #000000 !important;
            border: 1px solid #CCCCCC !important;
            box-shadow: none !important;
        }}

        /* hover/focus/active에서도 절대 검정으로 안 가게 */
        div.stButton > button:hover,
        div.stButton > button:active,
        div.stButton > button:focus,
        div[data-testid="stButton"] button:hover,
        div[data-testid="stButton"] button:active,
        div[data-testid="stButton"] button:focus,
        [data-testid^="baseButton-"]:hover,
        [data-testid^="baseButton-"]:active,
        [data-testid^="baseButton-"]:focus {{
            background: #FFFFFF !important;
            background-color: #FFFFFF !important;
            color: #000000 !important;
        }}

        /* st.info(stAlert) 텍스트 검정 */
        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] span {{
            color: #000000 !important;
        }}
    </style>
    """, unsafe_allow_html=True)



def main():
    st.set_page_config(
        page_title="한양대학교 학사관리 챗봇",
        page_icon="🦁",
        layout="centered"
    )

    apply_custom_styles()

    # 세션 초기화
    if "chat_session" not in st.session_state:
        try:
            st.session_state.chat_session = ChatSession()
        except Exception as e:
            st.error(f"초기화 실패: {e}")
            st.stop()

    session = st.session_state.chat_session

    # 헤더 영역
    col1, col2 = st.columns([8, 1])
    with col1:
        st.markdown(f'<div class="main-header">🦁 한양대 학사관리 챗봇</div>', unsafe_allow_html=True)
    with col2:
        if st.button("new", help="새 대화 시작"):
            st.session_state.chat_session = ChatSession()
            st.rerun()


    # 채팅 영역
    chat_container = st.container()

    # 1. 채팅 기록 표시
    with chat_container:
        if not session.history:
            st.info("궁금한 내용을 입력창에 남겨주세요!\n예시: 졸업 요건이 뭐야? / 장학금 신청 기간 알려줘")

        for msg in session.history:
            role = msg["role"]
            content = msg["content"]
            avatar = "👤" if role == "user" else "🦁"
            
            with st.chat_message(role, avatar=avatar):
                st.markdown(content)

    # 2. 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요..."):
        with chat_container:
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

        # 답변 생성
        with chat_container:
            with st.chat_message("assistant", avatar="🦁"):
                message_placeholder = st.empty()
                with st.spinner("답변을 작성 중입니다..."):
                    try:
                        result = session.ask(question=prompt)
                        answer = result.get("answer", "죄송합니다. 답변을 생성하지 못했습니다.")
                        
                        message_placeholder.markdown(answer)
                        
                    except Exception as e:
                        message_placeholder.error(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()