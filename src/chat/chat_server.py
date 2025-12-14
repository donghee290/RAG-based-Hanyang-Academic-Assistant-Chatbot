import os
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from langchain_chroma import Chroma

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.searching import load_vectorstore_openai
from src.rag.rag_pipeline import retrieve_chunks_for_rag, build_context_text
from src.rag.keyword_mapping import SOURCE_TITLE_KEYWORDS

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_source_title_by_keywords(question: str) -> Optional[str]:
    """
    질문에서 키워드를 추출하여 관련 source_title을 찾는 함수
    
    Args:
        question: 사용자 질문
        
    Returns:
        매칭된 source_title 또는 None
    """
    question_lower = question.lower()
    
    # 각 source_title의 키워드가 질문에 포함되는지 확인
    # 더 구체적인 키워드(긴 키워드)를 우선적으로 매칭
    matched_titles = []
    
    for title, keywords in SOURCE_TITLE_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in question_lower:
                # 키워드 길이를 우선순위로 사용 (더 긴 키워드가 더 구체적)
                matched_titles.append((title, len(keyword)))
                break  # 한 source_title에 대해 하나의 키워드만 매칭되면 충분
    
    if not matched_titles:
        return None
    
    # 우선순위: 1) 더 긴 키워드 매칭, 2) 첫 번째 매칭
    matched_titles.sort(key=lambda x: x[1], reverse=True)
    return matched_titles[0][0]


def call_openai_llm(
    context: str,
    question: str,
    model: str = "gpt-4o-mini",
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    OpenAI LLM을 호출해서 답변 생성 (멀티턴 history 지원)
    history: [{"role": "user"|"assistant", "content": "..."}] 형식
    """
    system_prompt = ( 
        "당신은 한양대학교 학사조교입니다. "
        "아래에 제공된 '참고 문서' 내용을 바탕으로 우선적으로 답변해 주세요. "
        "참고 문서에 해당 질문의 정답이 명확히 없더라도, "
        "관련된 정보(과목 개요, 학습목표, 주차별 계획, 수업 시간, 담당교수 등)가 있으면 그것을 정리해서 최대한 도움 되는 답변을 해 주세요. "
        "정말로 아무 관련 정보도 없을 때만 "
        "'해당 내용은 제공된 학사 안내 문서에서 찾을 수 없습니다.'라고 답해 주세요."
        "답변할 때 참고 문서의 모든 내용을 한꺼번에 나열하지 마세요. "
        "먼저 사용자의 질문과 직접적으로 연결된 핵심 정보만 2~4개 포인트로 간단히 요약해 제공하세요. "
        "그 다음, '더 자세한 정보가 필요하신가요?'와 같이 후속 질문을 유도하고, "
        "사용자가 추가 설명을 요청할 때에만 문서의 나머지 세부 내용을 단계적으로 제공하세요. "
        "이미 제공한 정보를 다시 반복하지 마세요. "
        "일상 대화(감사, 인사, 감탄사 등)는 검색을 무시하고 자연스럽게 응답하세요."
        "사용자의 요청에 대한 답변 이후, 당신의 추론 과정(문서 검색부터 대답을 생성하기까지의)을 상세하게 표시하세요."
    )

    # 이번 턴의 유저 메시지 (RAG 컨텍스트 포함)
    user_content = (
        "다음은 학사 안내 참고 문서입니다.\n\n"
        f"{context}\n\n"
        "위 문서를 참고하여 아래 질문에 답변해 주세요.\n\n"
        f"질문: {question}"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    # 과거 대화 히스토리 추가 (컨텍스트는 매 턴마다 새로 붙이되, history에는 넣지 않는 게 좋음)
    if history:
        messages.extend(history)

    # 이번 턴의 질문
    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


def rag_answer_openai(
    question: str,
    vectorstore: Chroma,
    k: int = 5,
    source_title: Optional[str] = None,
    auto_extract_source_title: bool = True,
    model: str = "gpt-4o-mini",
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    질문 하나에 대해 RAG + OpenAI LLM으로 답변 생성
    - history: 기존 대화 내역 (user/assistant) 리스트
    - source_title: 필터링할 source_title (None이면 자동 추출 시도)
    - auto_extract_source_title: source_title이 None일 때 자동 추출 여부
    - 반환: 답변 + 사용된 청크 정보
    """

    # 0) source_title 자동 추출 (옵션)
    if auto_extract_source_title and source_title is None:
        source_title = extract_source_title_by_keywords(question)

    # 1) 검색용 쿼리 구성
    search_query = question

    # 직전 user 질문이 있으면 같이 붙여서 검색
    if history:
        last_user_utterances = [m["content"] for m in history if m["role"] == "user"]
        if last_user_utterances:
            last_user = last_user_utterances[-1]
            search_query = last_user + "\n" + question

    # 2) 상위 k개 청크 검색
    chunks = retrieve_chunks_for_rag(
        vectorstore=vectorstore,
        query=search_query,
        k=k,
        source_title=source_title,
    )

    # 3) 컨텍스트 텍스트 생성
    context_text = build_context_text(chunks, max_chars=3000)

    # 4) LLM 호출 (history 포함)
    answer = call_openai_llm(
        context=context_text,
        question=question,
        model=model,
        history=history,
    )

    return {
        "question": question,
        "answer": answer,
        "used_chunks": chunks,
    }


class ChatSession:
    """
    멀티턴 RAG 챗봇 세션 관리용 클래스
    - vectorstore 로딩
    - history 관리
    - ask() 한 번이 RAG + LLM 한 턴
    """

    def __init__(self, vectorstore: Optional[Chroma] = None):
        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            self.vectorstore = load_vectorstore_openai()
            if self.vectorstore is None:
                raise RuntimeError("OpenAI 벡터스토어 로드 실패: OPENAI_API_KEY 또는 Chroma DB 확인 요망")

        self.history: List[Dict[str, str]] = []

    def ask(
        self,
        question: str,
        k: int = 5,
        source_title: Optional[str] = None,
        auto_extract_source_title: bool = True,
        model: str = "gpt-4o-mini",
    ) -> Dict[str, Any]:
        """
        한 턴 대화 처리
        - 내부적으로 rag_answer_openai 호출
        - history 업데이트까지 포함
        """
        result = rag_answer_openai(
            question=question,
            vectorstore=self.vectorstore,
            k=k,
            source_title=source_title,
            auto_extract_source_title=auto_extract_source_title,
            model=model,
            history=self.history,
        )

        answer = result["answer"]

        # 히스토리에 이번 턴 user/assistant 추가
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return result