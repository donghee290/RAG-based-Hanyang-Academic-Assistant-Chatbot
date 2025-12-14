# -*- coding: utf-8 -*-
"""
임베딩 모델 비교 및 벡터 DB 생성
- TF-IDF 임베딩
- 한국어 BERT 임베딩
- OpenAI Embeddings
"""

import json
import os
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# LangChain import (최신 버전 경로)
try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    try:
        from langchain.embeddings.base import Embeddings
    except ImportError:
        # 기본 ABC 클래스로 대체 (최소한의 호환성)
        from abc import ABC, abstractmethod
        class Embeddings(ABC):
            @abstractmethod
            def embed_documents(self, texts):
                pass
            @abstractmethod
            def embed_query(self, text):
                pass

# Chroma import (최신 버전)
Chroma = None
try:
    # 최신 버전: langchain_community 사용
    from langchain_community.vectorstores import Chroma
except ImportError:
    Chroma = None


# TF-IDF 커스텀 Embeddings 클래스
class TFIDFEmbeddings(Embeddings):
    """TF-IDF 기반 임베딩 클래스"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.vectors = None
        self.texts = []
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 리스트를 임베딩"""
        self.texts = texts
        self.vectors = self.vectorizer.fit_transform(texts)
        # sparse matrix를 dense list로 변환
        return self.vectors.toarray().tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """쿼리 텍스트를 임베딩"""
        if self.vectors is None:
            raise ValueError("먼저 embed_documents를 호출하여 모델을 학습시켜야 합니다.")
        query_vector = self.vectorizer.transform([text])
        return query_vector.toarray()[0].tolist()


def load_all_chunks(file_path: str = "./data/processed/all_chunks.json") -> List[Dict[str, Any]]:
    """통합 청크 데이터 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    return chunks


def create_chroma_db_openai(
    chunks: List[Dict[str, Any]],
    persist_directory: str = "./vectorstores/chroma_db_openai"
):
    """OpenAI Embeddings로 Chroma 벡터 DB 생성"""
    
    if Chroma is None:
        print("[ERROR] Chroma를 import할 수 없습니다.")
        return None
    
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        try:
            from langchain.embeddings import OpenAIEmbeddings
        except ImportError:
            return None
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    
    # 텍스트와 메타데이터 준비
    texts = [chunk['text'] for chunk in chunks]
    metadatas = [
        {
            "chunk_id": chunk['chunk_id'],
            "data_source": chunk.get('data_source', 'unknown'),
            "source_category": chunk.get('source_category', ''),
            "source_title": chunk.get('source_title', ''),
            "source_url": chunk.get('source_url', ''),
        }
        for chunk in chunks
    ]
    
    # 기존 DB 삭제 (있는 경우)
    if os.path.exists(persist_directory):
        import shutil
        shutil.rmtree(persist_directory)
    
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_directory
    )
    
    return vectorstore


def create_chroma_db_bert(
    chunks: List[Dict[str, Any]],
    persist_directory: str = "./vectorstores/chroma_db_bert"
):
    """한국어 BERT Embeddings로 Chroma 벡터 DB 생성"""
    
    if Chroma is None:
        return None
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
        except ImportError:
            return None
    
    # 한국어 BERT 모델 선택
    # 옵션: 
    # - "jhgan/ko-sroberta-multitask" (442MB, 다운로드 시간 오래 걸림)
    # - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (더 작고 빠름)
    # - "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" (중간 크기)
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # 모델 로드 (재시도 로직)
    max_retries = 3
    embeddings = None
    
    for attempt in range(max_retries):
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(5)
            else:
                model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                try:
                    embeddings = HuggingFaceEmbeddings(
                        model_name=model_name,
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': True}
                    )
                except Exception:
                    return None
    
    if embeddings is None:
        return None
    
    # 텍스트와 메타데이터 준비
    texts = [chunk['text'] for chunk in chunks]
    metadatas = [
        {
            "chunk_id": chunk['chunk_id'],
            "data_source": chunk.get('data_source', 'unknown'),
            "source_category": chunk.get('source_category', ''),
            "source_title": chunk.get('source_title', ''),
            "source_url": chunk.get('source_url', ''),
        }
        for chunk in chunks
    ]
    
    # 기존 DB 삭제 (있는 경우)
    if os.path.exists(persist_directory):
        import shutil
        shutil.rmtree(persist_directory)
    
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_directory
    )
    
    return vectorstore


def create_chroma_db_tfidf(
    chunks: List[Dict[str, Any]],
    persist_directory: str = "./vectorstores/chroma_db_tfidf"
):
    """TF-IDF Embeddings로 Chroma 벡터 DB 생성"""
    
    if Chroma is None:
        print("[ERROR] Chroma를 import할 수 없습니다.")
        return None
    
    embeddings = TFIDFEmbeddings()
    
    # 텍스트와 메타데이터 준비
    texts = [chunk['text'] for chunk in chunks]
    metadatas = [
        {
            "chunk_id": chunk['chunk_id'],
            "data_source": chunk.get('data_source', 'unknown'),
            "source_category": chunk.get('source_category', ''),
            "source_title": chunk.get('source_title', ''),
            "source_url": chunk.get('source_url', ''),
        }
        for chunk in chunks
    ]
    
    # TF-IDF 벡터 생성
    vectors = embeddings.embed_documents(texts)
    
    # 기존 DB 삭제 (있는 경우)
    if os.path.exists(persist_directory):
        import shutil
        shutil.rmtree(persist_directory)
    
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_directory
    )
    
    return vectorstore


if __name__ == "__main__":
    chunks = load_all_chunks("./data/processed/all_chunks.json")
    
    try:
        create_chroma_db_tfidf(chunks, "./vectorstores/chroma_db_tfidf")
    except Exception as e:
        print(f"TF-IDF 생성 실패: {e}")
    
    try:
        create_chroma_db_bert(chunks, "./vectorstores/chroma_db_bert")
    except Exception as e:
        print(f"BERT 생성 실패: {e}")
    
    try:
        create_chroma_db_openai(chunks, "./vectorstores/chroma_db_openai")
    except Exception as e:
        print(f"OpenAI 생성 실패: {e}")

