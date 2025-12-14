# RAG-based Hanyang Academic Assistant Chatbot

한양대학교 학사 정보 및 강의 데이터를 기반으로 질문에 답변하는
RAG(Retrieval-Augmented Generation) 챗봇 프로젝트입니다.

## 📌 주요 기능
- **데이터 수집**: 한양대학교 홈페이지 및 강의, 강의계획서 데이터 크롤링
- **RAG 파이프라인**: 
    - 텍스트 청킹 및 임베딩(OpenAI Embeddings)
    - 벡터 저장소 구축(ChromaDB)
    - LLM 기반 답변 생성(GPT-4o-mini)
- **사용자 인터페이스**: 
    - CLI(Command Line Interface)
    - Web UI(Streamlit)
    - APP(fastAPI)

## 🛠️ 설치 및 설정 (Installation)

### 1. 환경 설정
Python 3.10+ 환경에서 실행을 권장합니다.

```bash
# 가상환경 생성(선택 사항)
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 2. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정
프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 아래 내용을 입력하세요.
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 🚀 실행 가이드 (Usage)

### 1. 웹 인터페이스 모드(GUI) 실행
Streamlit을 사용하여 웹 브라우저에서 챗봇을 사용할 수 있습니다.

```bash
streamlit run src/chat/web_ui.py
```
- 실행 후 브라우저가 자동으로 열립니다.(기본 주소: `http://localhost:8501`)

### 2. 터미널 모드(CLI) 실행
간단한 터미널 인터페이스로 챗봇을 테스트할 수 있습니다.

```bash
python src/main.py
```

### 3. APP 모드(fastAPI) 실행
외부 애플리케이션에서 HTTP 요청을 통해 챗봇을 사용할 수 있습니다.

```bash
python src/app.py
```

## 프로젝트 구조 (Directory Structure)
```
tm_ver3/
├── data/               # 크롤링된 원본 데이터(git ignored)
├── vectorstores/       # ChromaDB 벡터 저장소(git ignored)
├── results/            # 모델 평가 결과(git ignored)
├── src/
│   ├── crawling/       # 데이터 수집
│   ├── preprocessing/  # 데이터 전처리 및 로더
│   ├── embedding/      # 임베딩 생성 및 저장
│   ├── retrieval/      # 검색 로직
│   ├── rag/            # RAG 파이프라인 코어
│   ├── evaluation/     # 모델 성능 평가
│   ├── chat/           # 챗봇 서버 및 UI 로직
│   ├── main.py         # CLI 진입점
│   └── app.py          # fastAPI 서버 진입점
├── .gitignore
├── requirements.txt
└── README.md
```

## ⚠️ 주의사항
- `data`, `vectorstores` 폴더는 용량이 크거나 보안상의 이유로 `.gitignore`에 포함되어 있습니다.
- 처음 실행 시 벡터 데이터가 없다면 임베딩 생성 과정이 필요할 수 있습니다.(관련 스크립트: `src/embedding/embedding.py` 확인 필요)