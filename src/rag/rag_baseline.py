import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import json
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional

from langchain_chroma import Chroma
from openai import OpenAI

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.searching import load_vectorstore_openai

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# 검색(Retrieval)

def retrieve_chunks_for_rag(
    vectorstore: Chroma,
    query: str,
    k: int = 5,
    data_source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    RAG용 상위 k개 청크를 가져오는 함수
    - similarity_search_with_score를 써도 되고
    - similarity_search만 써도 무방
    """
    filter_dict = {"data_source": data_source} if data_source else None

    docs = vectorstore.similarity_search(
        query=query,
        k=k,
        filter=filter_dict
    )

    results = []
    for rank, doc in enumerate(docs, 1):
        results.append({
            "rank": rank,
            "chunk_id": doc.metadata.get("chunk_id", -1),
            "text": doc.page_content,
            "metadata": doc.metadata,
        })
    return results


def build_context_text(chunks: List[Dict[str, Any]], max_chars: int = 3000) -> str:
    parts = []
    current_len = 0

    for ch in chunks:
        md = ch.get("metadata", {}) or {}
        src = md.get("data_source", "")
        chunk_id = ch.get("chunk_id", -1)

        # 기본 prefix
        prefix = f"[chunk_id={chunk_id}, source={src}]\n"

        # syllabus 같은 교과목 청크일 때 메타데이터를 최대한 활용
        meta_lines = []
        if src == "syllabus":
            # 실제 all_chunks.json에 맞춰서 있는 필드만 붙도록 방어적으로 작성
            course_id = md.get("course_id") or md.get("course_code")
            course_name = md.get("course_name") or md.get("course_title")
            instructor = md.get("instructor") or md.get("professor") or md.get("professor_name")
            year = md.get("year")
            semester = md.get("semester")
            week = md.get("week") or md.get("week_no")
            week_title = md.get("week_title") or md.get("topic")

            if course_id or course_name:
                meta_lines.append(f"과목: {course_id or ''} {course_name or ''}".strip())
            if instructor:
                meta_lines.append(f"담당교수: {instructor}")
            if year or semester:
                meta_lines.append(f"개설학기: {year or ''} {semester or ''}".strip())
            if week or week_title:
                meta_lines.append(f"주차: {week or ''} {week_title or ''}".strip())

            # 만약 syllabus 전문을 별도 필드로 넣어놨다면 (있으면 쓰고, 없으면 무시)
            syllabus_text = md.get("syllabus_text") or md.get("weekly_plan") or md.get("plan_text")
            if syllabus_text:
                meta_lines.append(f"주차별 계획(요약): {syllabus_text}")

        # 공통적으로 쓸 수 있는 source_title 같은 필드도 있으면 붙이기
        source_title = md.get("source_title")
        if source_title:
            meta_lines.append(f"문서제목: {source_title}")

        # 메타데이터 블록 문자열
        meta_block = ""
        if meta_lines:
            meta_block = "\n".join(meta_lines) + "\n\n"

        # 원래 청크 본문
        content = ch.get("text", "").strip()
        full_text = prefix + meta_block + content + "\n\n"

        # 남은 공간 계산
        remaining_chars = max_chars - current_len
        if remaining_chars <= 0:
            break

        # 텍스트가 남은 공간보다 길면 잘라서 넣기 (Truncate)
        if len(full_text) > remaining_chars:
            if remaining_chars > len(prefix) + 50:
                parts.append(full_text[:remaining_chars] + "...(truncated)\n")
                current_len += len(full_text[:remaining_chars])
            break  # 공간 꽉 참
        else:
            parts.append(full_text)
            current_len += len(full_text)

    # 아무것도 추가되지 않았다면 첫 청크 일부라도 강제로 넣기
    if not parts and chunks:
        first = chunks[0]
        first_md = first.get("metadata", {}) or {}
        first_src = first_md.get("data_source", "")
        first_id = first.get("chunk_id", -1)
        first_text = first.get("text", "")[:max_chars]

        header = f"[Fallback chunk_id={first_id}, source={first_src}]\n"
        parts.append(header + first_text)

    return "".join(parts)


# LLM
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

# 전체 RAG 파이프라인 함수

def rag_answer_openai(
    question: str,
    vectorstore: Chroma,
    k: int = 5,
    data_source: Optional[str] = None,
    model: str = "gpt-4o-mini",
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    질문 하나에 대해 RAG + OpenAI LLM으로 답변 생성
    - history: 기존 대화 내역 (user/assistant) 리스트
    - 반환: 답변 + 사용된 청크 정보
    """

    # 0) 검색용 쿼리 구성
    search_query = question

    # 직전 user 질문이 있으면 같이 붙여서 검색
    if history:
        last_user_utterances = [m["content"] for m in history if m["role"] == "user"]
        if last_user_utterances:
            last_user = last_user_utterances[-1]
            search_query = last_user + "\n" + question

    # 1) 상위 k개 청크 검색
    chunks = retrieve_chunks_for_rag(
        vectorstore=vectorstore,
        query=search_query,
        k=k,
        data_source=data_source,
    )

    # 2) 컨텍스트 텍스트 생성
    context_text = build_context_text(chunks, max_chars=3000)

    # 3) LLM 호출 (history 포함)
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


if __name__ == "__main__":
    print("RAG + OpenAI LLM 베이스라인 (멀티턴 챗봇)")
    vectorstore = load_vectorstore_openai()
    if vectorstore is None:
        raise RuntimeError("OpenAI 벡터스토어 로드 실패: OPENAI_API_KEY 또는 Chroma DB 확인 요망")

    # 대화 히스토리
    history: List[Dict[str, str]] = []

    while True:
        try:
            question = input("\n질문을 입력하세요 (종료: exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not question:
            continue
        if question.lower() in ["exit", "quit", "q"]:
            print("종료합니다.")
            break

        # 이번 턴 RAG + LLM
        result = rag_answer_openai(
            question=question,
            vectorstore=vectorstore,
            k=5,
            data_source=None,
            model="gpt-4o-mini",
            history=history,
        )

        answer = result["answer"]

        print("\n[답변]")
        print(answer)
        print("\n[사용된 청크 ID들]")
        print([c["chunk_id"] for c in result["used_chunks"]])

        # 히스토리에 이번 턴 user/assistant 추가
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})