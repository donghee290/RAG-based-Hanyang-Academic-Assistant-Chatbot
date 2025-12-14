import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import time
import os
import json
import torch
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from openai import OpenAI
from huggingface_hub import login

from src.retrieval.searching import load_vectorstore_openai
from src.rag.rag_baseline import retrieve_chunks_for_rag, build_context_text

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# -----------------------------
# 환경 설정
# -----------------------------
load_dotenv()

MAX_QUESTIONS = 5
MAX_NEW_TOKENS = 160
CONTEXT_MAX_CHARS = 2000

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다 (.env 또는 환경변수 확인).")
client_openai = OpenAI(api_key=OPENAI_API_KEY)

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN이 설정되어 있지 않습니다 (.env 또는 환경변수 확인).")

login(token=HF_TOKEN)

device = -1

LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
QWEN_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

_llama_pipe = None
_qwen_pipe = None


def get_llama_pipeline():
    global _llama_pipe
    if _llama_pipe is None:
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_NAME,
            token=HF_TOKEN,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        _llama_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            device=device
        )
    return _llama_pipe


def get_qwen_pipeline():
    global _qwen_pipe
    if _qwen_pipe is None:
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            token=HF_TOKEN,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        _qwen_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            device=device
        )
    return _qwen_pipe


# -----------------------------
# LLM별 호출 함수들
# -----------------------------

def _build_prompt_for_instruct_model(context: str, question: str) -> str:
    system_prompt = (
        "당신은 한양대학교 학사조교입니다. "
        "아래에 제공된 '참고 문서' 내용을 바탕으로 우선적으로 답변해 주세요. "
        "참고 문서에 해당 질문의 정답이 명확히 없더라도, "
        "관련된 정보(과목 개요, 학습목표, 주차별 계획, 수업 시간, 담당교수 등)가 있으면 "
        "그것을 정리해서 최대한 도움 되는 답변을 해 주세요. "
        "정말로 아무 관련 정보도 없을 때만 "
        "'해당 내용은 제공된 학사 안내 문서에서 찾을 수 없습니다.'라고 답해 주세요."
    )

    user_content = (
        "다음은 학사 안내 참고 문서입니다.\n\n"
        f"{context}\n\n"
        "위 문서를 참고하여 아래 질문에 답변해 주세요.\n\n"
        f"질문: {question}\n"
    )

    prompt = f"System: {system_prompt}\n\nUser: {user_content}\nAssistant:"
    return prompt


def call_openai_llm(context: str, question: str, model: str = "gpt-4o-mini") -> str:
    system_prompt = (
        "당신은 한양대학교 학사조교입니다. "
        "아래에 제공된 '참고 문서' 내용을 바탕으로 우선적으로 답변해 주세요. "
        "참고 문서에 해당 질문의 정답이 명확히 없더라도, "
        "관련된 정보(과목 개요, 학습목표, 주차별 계획, 수업 시간, 담당교수 등)가 있으면 "
        "그것을 정리해서 최대한 도움 되는 답변을 해 주세요. "
        "정말로 아무 관련 정보도 없을 때만 "
        "'해당 내용은 제공된 학사 안내 문서에서 찾을 수 없습니다.'라고 답해 주세요."
    )

    user_content = (
        "다음은 학사 안내 참고 문서입니다.\n\n"
        f"{context}\n\n"
        "위 문서를 참고하여 아래 질문에 답변해 주세요.\n\n"
        f"질문: {question}"
    )

    resp = client_openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def call_llama_llm(context: str, question: str) -> str:
    pipe = get_llama_pipeline()
    prompt = _build_prompt_for_instruct_model(context, question)

    outputs = pipe(prompt)
    text = outputs[0]["generated_text"]
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()


def call_qwen_llm(context: str, question: str) -> str:
    pipe = get_qwen_pipeline()
    prompt = _build_prompt_for_instruct_model(context, question)

    outputs = pipe(prompt)
    text = outputs[0]["generated_text"]
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()


def generate_answer(model_name: str, context: str, question: str) -> str:
    if model_name == "openai":
        return call_openai_llm(context, question)
    elif model_name == "llama":
        return call_llama_llm(context, question)
    elif model_name == "qwen":
        return call_qwen_llm(context, question)
    else:
        raise ValueError(f"지원하지 않는 모델명: {model_name}")


# -----------------------------
# sBERT 기반 정량 평가
# -----------------------------

def load_sbert_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def cosine_similarity_sbert(
    model: SentenceTransformer,
    ref_answer: str,
    pred_answer: str
) -> float:
    emb_ref = model.encode(ref_answer, convert_to_tensor=True)
    emb_pred = model.encode(pred_answer, convert_to_tensor=True)
    sim = util.cos_sim(emb_ref, emb_pred).item()
    return float(sim)

def cosine_similarity_sbert_from_emb(
    model: SentenceTransformer,
    emb_ref,
    pred_answer: str
) -> float:
    emb_pred = model.encode(pred_answer, convert_to_tensor=True)
    sim = util.cos_sim(emb_ref, emb_pred).item()
    return float(sim)

# -----------------------------
# 메인: 평가 파이프라인
# -----------------------------

def evaluate_llms_on_dataset(
    vectorstore: Chroma,
    eval_path: str,
    k: int = 5,
    model_names: List[str] = None,
    output_path: str = ".results/llm_results/llm_comparison_results2.json",
    max_questions: Optional[int] = None,
    skip_completed: bool = True,   # 이미 수행된 모델은 스킵할지 여부
):
    if model_names is None:
        model_names = ["openai", "llama", "qwen"]

    # 1) 평가 데이터 로드
    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", data)

    # 질문 개수 제한 (CPU 환경용)
    if max_questions is not None and len(questions) > max_questions:
        questions = questions[:max_questions]
        print(f"[INFO] 질문 {len(questions)}개만 사용 (max_questions={max_questions})")

    # 2) 기존 결과가 있으면 불러와서 details를 머지
    existing_details_by_id: Dict[Any, Dict[str, Any]] = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                prev_results = json.load(f)
            for item in prev_results.get("details", []):
                existing_details_by_id[item["id"]] = item
            print(f"[INFO] 기존 결과 {len(existing_details_by_id)}개 로드됨 (output_path={output_path})")
        except Exception as e:
            print(f"[WARN] 기존 결과 로드 실패, 새로 생성합니다: {e}")

    sbert = load_sbert_model()

    all_details: List[Dict[str, Any]] = []
    stats: Dict[str, List[float]] = {m: [] for m in model_names}

    for idx, q in enumerate(questions, start=1):
        q_id = q.get("id", idx)
        question_text = q["question"]
        ref_answer = q.get("answer") or q.get("ground_truth_answer", "")

        if not ref_answer:
            print(f"[WARN] id={q_id}: reference answer가 없어 스킵합니다.")
            continue

        raw_data_source = q.get("data_source")
        if raw_data_source in (None, "", "chunks"):
            data_source = None
        else:
            data_source = raw_data_source

        # 3) RAG 검색 (한 번만)
        chunks = retrieve_chunks_for_rag(
            vectorstore=vectorstore,
            query=question_text,
            k=k,
            data_source=data_source,
        )

        if not chunks:
            print(f"[WARN] id={q_id}: 검색 결과 없음 (질문='{question_text[:20]}...')")
            context = ""
        else:
            # 컨텍스트 길이 줄이기 (위에서 정의한 상수 사용해도 됨)
            context = build_context_text(chunks, max_chars=CONTEXT_MAX_CHARS)

        # 질문당 ref 임베딩 한 번만
        ref_emb = sbert.encode(ref_answer, convert_to_tensor=True)

        # 기존에 이 질문 id에 대한 결과가 있으면 가져옴
        if q_id in existing_details_by_id:
            item_result = existing_details_by_id[q_id]
        else:
            item_result = {
                "id": q_id,
                "question": question_text,
                "reference_answer": ref_answer,
                "data_source": data_source,
                "results": {},
            }

        # 4) 각 모델별로 아직 없는 결과만 생성
        for model_name in model_names:
            # 이미 이전 실행에서 이 모델 결과가 있고, skip_completed=True면 스킵
            if skip_completed and model_name in item_result["results"]:
                sim = item_result["results"][model_name].get("similarity")
                if sim is not None:
                    stats[model_name].append(sim)
                continue

            try:
                # 여기서부터 시간 측정 시작
                start = time.perf_counter()
                answer = generate_answer(model_name, context, question_text)
                elapsed = time.perf_counter() - start  # 응답 생성에 걸린 시간(초)
                # 여기까지 시간 측정

                sim = cosine_similarity_sbert_from_emb(sbert, ref_emb, answer)
            except Exception as e:
                print(f"[ERROR] 모델 {model_name} 처리 중 오류 (id={q_id}): {e}")
                answer = ""
                sim = None
                elapsed = None

            item_result["results"][model_name] = {
                "answer": answer,
                "similarity": sim,
                "latency_sec": elapsed,  # 응답 시간까지 결과에 포함
            }

            if sim is not None:
                stats[model_name].append(sim)


        all_details.append(item_result)
        print(f"[INFO] id={q_id} 처리 완료")

    # 5) summary 재계산 (새로 생성된 all_details 기반)
    summary: Dict[str, Dict[str, Any]] = {}
    # 모델 목록은 details 안에 실제로 등장한 모델 기준으로 다시 뽑아도 됨
    all_model_names = set()
    for item in all_details:
        all_model_names.update(item.get("results", {}).keys())

    for m in all_model_names:
        sims = []
        latencies = []
        for item in all_details:
            r = item.get("results", {}).get(m)
            if r is None:
                continue

            sim = r.get("similarity")
            if sim is not None:
                sims.append(sim)

            lat = r.get("latency_sec")
            if lat is not None:
                latencies.append(lat)

        if sims:
            summary[m] = {
                "mean_similarity": float(sum(sims) / len(sims)),
                "mean_latency_sec": float(sum(latencies) / len(latencies)) if latencies else None,
                "num_samples": len(sims),
            }
        else:
            summary[m] = {
                "mean_similarity": None,
                "mean_latency_sec": None,
                "num_samples": 0,
            }


    result_obj = {
        "summary": summary,
        "details": all_details,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_obj, f, ensure_ascii=False, indent=2)

    print("\n=== LLM 비교 요약 ===")
    for m, info in summary.items():
        print(
            f"- {m}: mean_similarity={info['mean_similarity']}, "
            f"mean_latency_sec={info['mean_latency_sec']}, "
            f"n={info['num_samples']}"
        )

    print(f"\n전체 결과 저장 완료 → {output_path}")


def get_next_indexed_paths(
    eval_dir: str,
    eval_prefix: str,
    result_dir: str,
    result_prefix: str,
    ext: str = ".json"
):
    idx = 1
    while True:
        eval_path = os.path.join(eval_dir, f"{eval_prefix}{idx}{ext}")
        result_path = os.path.join(result_dir, f"{result_prefix}{idx}{ext}")

        # 하나라도 존재하면 다음 번호
        if os.path.exists(eval_path) or os.path.exists(result_path):
            idx += 1
            continue

        return eval_path, result_path, idx


if __name__ == "__main__":
    print("LLM 비교 실험 시작 (OpenAI / LLaMA / Qwen)")
    vectorstore = load_vectorstore_openai()
    if vectorstore is None:
        raise RuntimeError("OpenAI 벡터스토어 로드 실패")

    eval_path, output_path, run_idx = get_next_indexed_paths(
        eval_dir="./data/evaluation",
        eval_prefix="evaluation_qa_llm",
        result_dir="./results/llm_results",
        result_prefix="llm_comparison_results"
    )
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"평가셋 파일이 없습니다: {eval_path}")

    print(f"[INFO] 평가 실행 번호: {run_idx}")
    print(f"[INFO] eval_path = {eval_path}")
    print(f"[INFO] output_path = {output_path}")

    evaluate_llms_on_dataset(
        vectorstore=vectorstore,
        eval_path=eval_path,
        k=5,
        # model_names=["openai", "qwen"],
        # model_names=["llama"],
        model_names=["openai", "llama", "qwen"],
        output_path=output_path,
        max_questions=MAX_QUESTIONS
    )