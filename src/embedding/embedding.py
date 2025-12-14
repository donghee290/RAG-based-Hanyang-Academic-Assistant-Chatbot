# -*- coding: utf-8 -*-
"""
all_chunks.json 기준 OpenAI 임베딩 → Chroma 벡터 DB 생성 스크립트
"""

import json
import os
import shutil
import traceback
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    # langchain 구버전 호환
    from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")


def load_all_chunks(file_path: str = "./data/processed/all_chunks.json") -> List[Dict[str, Any]]:
    """통합 청크 데이터 로드"""
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks


def create_chroma_db_openai(
    chunks: List[Dict[str, Any]],
    persist_directory: str = "./vectorstores/chroma_db_openai",
    model: str = "text-embedding-3-small",
):
    """OpenAI Embeddings로 Chroma 벡터 DB 생성"""

    embeddings = OpenAIEmbeddings(
        model=model,
        openai_api_key=api_key,
    )

    texts = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        # 1) 기본 text
        base_text = chunk.get("text", "")

        # 2) syllabus_text 같이 붙이기 (최상위 키에 있음)
        syllabus_text = chunk.get("syllabus_text", "")
        if syllabus_text:
            text = base_text + "\n\n[강의계획서 상세]\n" + syllabus_text
        else:
            text = base_text

        texts.append(text)

        # 메타데이터 구성
        md = dict(chunk)     # 원본 복사
        md.pop("text", None) # text는 메타데이터에서 제거
        # syllabus_text는 메타데이터에 남겨두고 싶으면 pop 하지 말 것
        # md.pop("syllabus_text", None)  # <- 필요하면 이 줄은 빼고 그대로 둬도 됨

        if "chunk_id" not in md:
            md["chunk_id"] = i

        md.setdefault("data_source", "unknown")

        metadatas.append(md)

    # 기존 DB 삭제
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_directory,
    )

    vectorstore.persist()
    return vectorstore


if __name__ == "__main__":

    print("all_chunks.json → OpenAI 임베딩 → Chroma DB(chroma_db_openai) 생성")
    chunks = load_all_chunks("./data/processed/all_chunks.json")
    print(f"로드된 청크 개수: {len(chunks)}")

    try:
        vs = create_chroma_db_openai(
            chunks,
            persist_directory="./vectorstores/chroma_db_openai",
            model="text-embedding-3-small",
        )
        print("벡터스토어 생성 및 저장 완료.")
    except Exception as e:
        print("[ERROR] 벡터스토어 생성 중 예외 발생:")
        traceback.print_exc()