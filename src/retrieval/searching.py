# -*- coding: utf-8 -*-
"""
검색 모듈 (OpenAI 임베딩 + Chroma 전용)
- chroma_db_openai 벡터스토어 로드
- 단일 쿼리 검색 함수 제공
"""

import os
from typing import List, Dict, Any, Optional

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma  # 하위버전 호환


def load_vectorstore_openai(
    persist_directory: str = "./vectorstores/chroma_db_openai",
):
    """
    OpenAI 임베딩으로 생성된 Chroma 벡터스토어 로드

    전제:
    - embedding.py에서 OpenAIEmbeddings("text-embedding-3-small")로
      Chroma.from_texts(..., persist_directory="./vectorstores/chroma_db_openai") 생성 완료 상태
    """
    # OpenAI 임베딩 클래스 로드
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        try:
            from langchain.embeddings import OpenAIEmbeddings
        except ImportError:
            print("[ERROR] OpenAIEmbeddings 임포트 실패")
            return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY가 설정되어 있지 않습니다.")
        return None

    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
        )

        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
        return vectorstore

    except Exception as e:
        print(f"[ERROR] OpenAI 벡터 DB 로드 실패: {e}")
        return None


def search_openai(
    query: str,
    vectorstore: Chroma,
    k: int = 5,
    source_title: Optional[str] = None,
    preview_chars: int = 200,
) -> List[Dict[str, Any]]:
    """
    OpenAI 기반 Chroma에서 단일 쿼리 검색
    - similarity_search_with_score 사용
    - source_title 메타데이터 기준 필터링 가능
    - RAG / 디버깅에서 그대로 재사용 가능
    
    source_title별 처리:
    - source_title이 있으면: 2단계 검색 + 재정렬 (일치 우선, 없는 것도 포함)
    - source_title이 없으면: 필터 없이 전체 검색
    """
    if vectorstore is None:
        raise ValueError("vectorstore가 None입니다. load_vectorstore_openai() 결과를 확인하세요.")

    try:
        if source_title:
            # source_title 필터링: 2단계 검색 + 재정렬
            expanded_k = k * 2
            search_results = vectorstore.similarity_search_with_score(
                query=query,
                k=expanded_k,
                filter=None  # 필터 없이 전체 검색
            )
            
            # source_title 기준으로 재정렬
            matched = []  # source_title 일치
            no_title = []  # source_title 없음
            
            for doc, score in search_results:
                doc_title = doc.metadata.get("source_title", "")
                if doc_title == source_title:
                    matched.append((doc, score))
                elif not doc_title:  # 빈 문자열 또는 필드 없음
                    no_title.append((doc, score))
            
            # 일치하는 것 먼저, 그 다음 없는 것, 유사도 점수 순으로 정렬
            all_results = sorted(matched, key=lambda x: x[1]) + \
                         sorted(no_title, key=lambda x: x[1])
            search_results = [(doc, score) for doc, score in all_results[:k]]
        else:
            # source_title 없음: 필터 없이 전체 검색
            search_results = vectorstore.similarity_search_with_score(
                query=query,
                k=k,
            )
    except Exception as e:
        print(f"[ERROR] 검색 실패: {e}")
        return []

    formatted: List[Dict[str, Any]] = []
    for rank, (doc, score) in enumerate(search_results, 1):
        text = doc.page_content
        if len(text) > preview_chars:
            preview = text[:preview_chars] + "..."
        else:
            preview = text

        formatted.append({
            "rank": rank,
            "score": float(score),
            "chunk_id": doc.metadata.get("chunk_id", -1),
            "text_preview": preview,
            "full_text": text,
            "metadata": doc.metadata,
        })

    return formatted