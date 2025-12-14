from typing import Optional, Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from src.chat.chat_server import ChatSession


# FastAPI 인스턴스 생성
app = FastAPI(
    title="HY Academic Chatbot API",
    description="한양대 학사관리 챗봇 API",
    version="0.1.0",
)

# 서버 전체에서 공유할 ChatSession (하나만 생성해서 재사용)
session = ChatSession()


# 1) 요청 스키마
class ChatRequest(BaseModel):
    question: str


# 2) 응답 스키마
class ChatResponse(BaseModel):
    answer: str
    meta: Optional[Dict[str, Any]] = None


# 3) 헬스 체크 (옵션)
@app.get("/health")
def health_check():
    return {"status": "ok"}


# 4) 실제 채팅 엔드포인트
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    한양대 학사관리 챗봇 파이프라인을 한 번 실행하는 HTTP 엔드포인트
    """
    # 기존 CLI에서 하던 것과 거의 동일한 호출
    result: Dict[str, Any] = session.ask(
        question=req.question,
        k=5,
        data_source=None,
        model="gpt-4o-mini",
    )

    answer = result.get("answer", "")
    meta = {k: v for k, v in result.items() if k != "answer"}

    return ChatResponse(answer=answer, meta=meta)