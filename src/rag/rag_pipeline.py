import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import Dict, Any, List, Optional
from langchain_chroma import Chroma


def retrieve_chunks_for_rag(
    vectorstore: Chroma,
    query: str,
    k: int = 5,
    source_title: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    RAG용 상위 k개 청크를 가져오는 함수
    - similarity_search_with_score를 써도 되고
    - similarity_search만 써도 무방
    
    source_title별 처리:
    - source_title이 있으면: 2단계 검색 + 재정렬 (일치 우선, 없는 것도 포함)
    - source_title이 없으면: 필터 없이 전체 검색 (초기 로직)
    """
    if source_title:
        # source_title 필터링: 2단계 검색 + 재정렬
        expanded_k = k * 2
        docs_with_scores = vectorstore.similarity_search_with_score(
            query=query,
            k=expanded_k,
            filter=None  # 필터 없이 전체 검색
        )
        
        # source_title 기준으로 재정렬
        matched = []  # source_title 일치
        no_title = []  # source_title 없음
        
        for doc, score in docs_with_scores:
            doc_title = doc.metadata.get("source_title", "")
            if doc_title == source_title:
                matched.append((doc, score))
            elif not doc_title:  # 빈 문자열 또는 필드 없음
                no_title.append((doc, score))
        
        # 일치하는 것 먼저, 그 다음 없는 것, 유사도 점수 순으로 정렬
        # 점수가 낮을수록 유사도가 높음 (distance 기반)
        all_results = sorted(matched, key=lambda x: x[1]) + \
                     sorted(no_title, key=lambda x: x[1])
        docs = [doc for doc, _ in all_results[:k]]
    else:
        # source_title 없음: 필터 없이 전체 검색 (초기 로직)
        docs = vectorstore.similarity_search(
            query=query,
            k=k,
            filter=None
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