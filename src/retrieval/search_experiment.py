# -*- coding: utf-8 -*-
"""
검색 시스템 및 성능 평가 모듈
- 세 가지 임베딩 모델별 검색
- 평가 메트릭 계산
- 성능 비교 리포트 생성
"""

import json
import os
from typing import List, Dict, Any, Optional
from collections import defaultdict

# LangChain import
try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    try:
        from langchain.embeddings.base import Embeddings
    except ImportError:
        from abc import ABC, abstractmethod
        class Embeddings(ABC):
            @abstractmethod
            def embed_documents(self, texts):
                pass
            @abstractmethod
            def embed_query(self, text):
                pass

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

# TF-IDF 임베딩 클래스 (embedding_comparison.py에서 복사)
from sklearn.feature_extraction.text import TfidfVectorizer

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
        return self.vectors.toarray().tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """쿼리 텍스트를 임베딩"""
        if self.vectors is None:
            raise ValueError("먼저 embed_documents를 호출하여 모델을 학습시켜야 합니다.")
        query_vector = self.vectorizer.transform([text])
        return query_vector.toarray()[0].tolist()


def load_vectorstore_tfidf(persist_directory: str = "./vectorstores/chroma_db_tfidf"):
    """TF-IDF 벡터 DB 로드"""
    try:
        # 원본 텍스트 데이터 로드 (vectorizer 학습을 위해 필수)
        with open("./data/processed/all_chunks.json", 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        texts = [chunk['text'] for chunk in chunks]
        
        # TF-IDF 임베딩 객체 생성 및 학습
        embeddings = TFIDFEmbeddings()
        embeddings.embed_documents(texts)  # vectorizer 학습 (핵심!)
        
        # 벡터 DB 로드
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        return vectorstore
    except Exception as e:
        print(f"TF-IDF 벡터 DB 로드 실패: {e}")
        return None


def load_vectorstore_bert(persist_directory: str = "./vectorstores/chroma_db_bert"):
    """BERT 벡터 DB 로드"""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
        except ImportError:
            return None
    
    try:
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        return vectorstore
    except Exception as e:
        print(f"BERT 벡터 DB 로드 실패: {e}")
        return None


def load_vectorstore_openai(persist_directory: str = "./vectorstores/chroma_db_openai"):
    """OpenAI 벡터 DB 로드"""
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        try:
            from langchain.embeddings import OpenAIEmbeddings
        except ImportError:
            return None
    
    # OpenAI API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = ""
    
    if not api_key:
        return None
    
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        return vectorstore
    except Exception as e:
        print(f"OpenAI 벡터 DB 로드 실패: {e}")
        return None


def search_single_query(
    query: str,
    vectorstores: Dict[str, Any],
    k: int = 5,
    data_source: str = None
) -> Dict[str, List[Dict[str, Any]]]:

    results = {}
    
    # 필터 설정
    filter_dict = None
    if data_source:
        filter_dict = {"data_source": data_source}
    
    for model_name, vectorstore in vectorstores.items():
        if vectorstore is None:
            continue
        
        try:
            # 유사도 점수와 함께 검색 (필터링 적용)
            if filter_dict:
                search_results = vectorstore.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            else:
                search_results = vectorstore.similarity_search_with_score(
                    query=query,
                    k=k
                )
            
            # 결과 포맷팅
            formatted_results = []
            for rank, (doc, score) in enumerate(search_results, 1):
                formatted_results.append({
                    "rank": rank,
                    "score": float(score),
                    "chunk_id": doc.metadata.get('chunk_id', -1),
                    "text": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                })
            
            results[model_name] = formatted_results
            
        except Exception as e:
            print(f"{model_name} 검색 실패: {e}")
            results[model_name] = []
    
    return results


def calculate_topk_accuracy(
    search_results: List[Dict[str, Any]],
    ground_truth_chunk_ids: List[int],
    k: int
) -> float:

    if not search_results or not ground_truth_chunk_ids:
        return 0.0
    
    # Top-K 결과의 청크 ID 추출
    topk_chunk_ids = [result['chunk_id'] for result in search_results[:k]]
    
    # 정답이 Top-K에 포함되는지 확인
    for gt_id in ground_truth_chunk_ids:
        if gt_id in topk_chunk_ids:
            return 1.0
    
    return 0.0


def calculate_mrr(
    search_results: List[Dict[str, Any]],
    ground_truth_chunk_ids: List[int]
) -> float:

    if not search_results or not ground_truth_chunk_ids:
        return 0.0
    
    # 정답이 나타나는 첫 번째 순위 찾기
    for rank, result in enumerate(search_results, 1):
        if result['chunk_id'] in ground_truth_chunk_ids:
            return 1.0 / rank
    
    return 0.0


def calculate_average_score(search_results: List[Dict[str, Any]], k: int = 5) -> float:
    """평균 유사도 점수 계산"""
    if not search_results:
        return 0.0
    
    topk_results = search_results[:k]
    scores = [result['score'] for result in topk_results]
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_models(
    evaluation_data: List[Dict[str, Any]],
    vectorstores: Dict[str, Any],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, Any]:

    # 결과 저장용
    all_results = []
    model_metrics = defaultdict(lambda: {
        'topk_accuracy': defaultdict(list),
        'mrr': [],
        'average_score': []
    })
    
    # 각 질문에 대해 검색 및 평가
    for idx, question_data in enumerate(evaluation_data, 1):
        question = question_data['question']
        ground_truth = question_data['ground_truth_chunk_ids']
        
        # 질문의 data_source 가져오기
        question_data_source = question_data.get('data_source')
        
        # 모든 모델로 검색 (data_source 필터링 적용)
        search_results = search_single_query(
            query=question,
            vectorstores=vectorstores,
            k=max(k_values),
            data_source=question_data_source
        )
        
        # 각 모델별로 메트릭 계산
        question_result = {
            "question_id": question_data['id'],
            "question": question,
            "category": question_data.get('category', ''),
            "data_source": question_data.get('data_source', ''),
            "ground_truth_chunk_ids": ground_truth,
            "model_results": {}
        }
        
        for model_name, results in search_results.items():
            # Top-K 정확도 계산
            topk_acc = {}
            for k in k_values:
                acc = calculate_topk_accuracy(results, ground_truth, k)
                topk_acc[f"top{k}"] = acc
                model_metrics[model_name]['topk_accuracy'][f"top{k}"].append(acc)
            
            # MRR 계산
            mrr = calculate_mrr(results, ground_truth)
            model_metrics[model_name]['mrr'].append(mrr)
            
            # 평균 점수
            avg_score = calculate_average_score(results, k=max(k_values))
            model_metrics[model_name]['average_score'].append(avg_score)
            
            question_result["model_results"][model_name] = {
                "topk_accuracy": topk_acc,
                "mrr": mrr,
                "average_score": avg_score,
                "search_results": results
            }
        
        all_results.append(question_result)
    
    # 전체 평균 메트릭 계산
    final_metrics = {}
    for model_name, metrics in model_metrics.items():
        final_metrics[model_name] = {
            "topk_accuracy": {
                k: sum(metrics['topk_accuracy'][k]) / len(metrics['topk_accuracy'][k])
                for k in metrics['topk_accuracy'].keys()
            },
            "mrr": sum(metrics['mrr']) / len(metrics['mrr']) if metrics['mrr'] else 0.0,
            "average_score": sum(metrics['average_score']) / len(metrics['average_score']) if metrics['average_score'] else 0.0
        }
    
    return {
        "summary": final_metrics,
        "detailed_results": all_results,
        "total_questions": len(evaluation_data)
    }


def save_results(
    results: Dict[str, Any],
    output_dir: str = "./results/search_results"
):
    """검색 결과 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 상세 결과 저장
    detailed_path = os.path.join(output_dir, "search_results.json")
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 요약 결과 저장
    summary_path = os.path.join(output_dir, "comparison_results.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": results['summary'],
            "total_questions": results['total_questions']
        }, f, ensure_ascii=False, indent=2)
    
    # 텍스트 리포트 생성
    report_path = os.path.join(output_dir, "search_report.txt")
    generate_text_report(results, report_path)


def generate_text_report(results: Dict[str, Any], filepath: str):
    """텍스트 형식 리포트 생성"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("임베딩 모델별 검색 성능 비교 리포트\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"총 평가 질문 수: {results['total_questions']}\n\n")
        
        # 요약 메트릭
        f.write("=" * 80 + "\n")
        f.write("모델별 성능 요약\n")
        f.write("=" * 80 + "\n\n")
        
        summary = results['summary']
        for model_name, metrics in summary.items():
            f.write(f"\n[{model_name.upper()}]\n")
            f.write("-" * 80 + "\n")
            
            # Top-K 정확도
            f.write("Top-K 정확도:\n")
            for k, acc in sorted(metrics['topk_accuracy'].items(), key=lambda x: int(x[0][3:])):
                f.write(f"  {k.upper()}: {acc:.4f} ({acc*100:.2f}%)\n")
            
            # MRR
            f.write(f"MRR: {metrics['mrr']:.4f}\n")
            
            # 평균 점수
            f.write(f"평균 유사도 점수: {metrics['average_score']:.4f}\n")
        
        # 모델 비교
        f.write("\n" + "=" * 80 + "\n")
        f.write("모델 비교 (Top-5 정확도 기준)\n")
        f.write("=" * 80 + "\n\n")
        
        models_sorted = sorted(
            summary.items(),
            key=lambda x: x[1]['topk_accuracy'].get('top5', 0),
            reverse=True
        )
        
        for rank, (model_name, metrics) in enumerate(models_sorted, 1):
            top5_acc = metrics['topk_accuracy'].get('top5', 0)
            f.write(f"{rank}. {model_name.upper()}: {top5_acc:.4f} ({top5_acc*100:.2f}%)\n")


if __name__ == "__main__":
    vectorstores = {
        "tfidf": load_vectorstore_tfidf(),
        "bert": load_vectorstore_bert(),
        "openai": load_vectorstore_openai()
    }
    
    vectorstores = {k: v for k, v in vectorstores.items() if v is not None}
    
    if not vectorstores:
        print("사용 가능한 벡터 DB가 없습니다.")
        exit(1)
    
    try:
        with open("./data/evaluation/evaluation_qa.json", 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        evaluation_questions = eval_data.get('questions', [])
    except Exception as e:
        print(f"평가 데이터 로드 실패: {e}")
        exit(1)
    
    results = evaluate_models(
        evaluation_data=evaluation_questions,
        vectorstores=vectorstores,
        k_values=[1, 3, 5, 10]
    )
    
    save_results(results)
    
    summary = results['summary']
    for model_name, metrics in summary.items():
        top5 = metrics['topk_accuracy'].get('top5', 0)
        mrr = metrics['mrr']
        print(f"{model_name.upper()}: Top-5 정확도={top5:.4f}, MRR={mrr:.4f}")

