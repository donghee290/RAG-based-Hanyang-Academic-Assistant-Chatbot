# -*- coding: utf-8 -*-
"""
데이터 로더 및 청킹 모듈
- chunks.json 로드
- courses_2025_2.csv 처리 및 청킹
- syllabus_2025_2.json 처리 및 청킹
- 통합 데이터셋 생성
"""

import json
import pandas as pd
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_chunks_json(file_path: str = "./data/processed/chunks.json") -> List[Dict[str, Any]]:
    """기존 chunks.json 파일 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    return chunks


def create_course_text(row: pd.Series) -> str:
    """강좌 정보를 검색 가능한 텍스트로 변환"""
    # 결측치 처리
    def safe_get(key, default=""):
        value = row.get(key, default)
        if pd.isna(value) or value == "":
            return default
        return str(value)
    
    text_parts: List[str] = []
    
    # 기본 정보
    if safe_get('수업번호'):
        text_parts.append(f"수업번호: {safe_get('수업번호')}")
    if safe_get('학수번호'):
        text_parts.append(f"학수번호: {safe_get('학수번호')}")
    if safe_get('교과목명'):
        text_parts.append(f"교과목명: {safe_get('교과목명')}")
    if safe_get('영문명'):
        text_parts.append(f"영문명: {safe_get('영문명')}")
    if safe_get('학점'):
        text_parts.append(f"학점: {safe_get('학점')}학점")
    if safe_get('이수구분'):
        text_parts.append(f"이수구분: {safe_get('이수구분')}")
    if safe_get('이수단위'):
        text_parts.append(f"이수단위: {safe_get('이수단위')}")
    if safe_get('학년'):
        text_parts.append(f"학년: {safe_get('학년')}학년")
    if safe_get('개설학기명'):
        text_parts.append(f"개설학기: {safe_get('개설학기명')}")
    if safe_get('교강사'):
        text_parts.append(f"교강사: {safe_get('교강사')}")
    if safe_get('소속학과'):
        text_parts.append(f"소속학과: {safe_get('소속학과')}")
    if safe_get('관장학과'):
        text_parts.append(f"관장학과: {safe_get('관장학과')}")
    if safe_get('강좌유형'):
        text_parts.append(f"강좌유형: {safe_get('강좌유형')}")
    if safe_get('정원'):
        text_parts.append(f"정원: {safe_get('정원')}")
    if safe_get('6C핵심역량'):
        text_parts.append(f"6C핵심역량: {safe_get('6C핵심역량')}")
    
    day = safe_get('요일')
    start_period = safe_get('시작교시')
    end_period = safe_get('종료교시')
    start_time = safe_get('시작시간')
    end_time = safe_get('종료시간')

    time_str_parts = []
    if day:
        time_str_parts.append(day)
    if start_period and end_period:
        time_str_parts.append(f"{start_period}교시~{end_period}교시")
    if start_time and end_time:
        time_str_parts.append(f"({start_time}~{end_time})")
    
    if time_str_parts:
        text_parts.append("수업시간: " + " ".join(time_str_parts))
    
    if safe_get('강의실'):
        text_parts.append(f"강의실: {safe_get('강의실')}")

    return "\n".join(text_parts)


###################### syllabus chunk ##########################################

def load_syllabus_json(path: str = "./data/raw/syllabus_2025_2.json") -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def deduplicate_syllabus(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    같은 suupNo(수업번호)가 여러 개 있을 수 있으므로,
    '내용이 더 풍부한' 레코드를 1개만 남긴다.
    """
    best_by_key: Dict[str, Dict[str, Any]] = {}

    for rec in records:
        haksu_no = str(rec.get("haksuNo", "")).strip()
        suup_no = str(rec.get("suupNo", "")).strip()
        key = haksu_no or suup_no
        if not key:
            continue

        # 내용 풍부도 단순 점수: overview + objectives + weekly_plan 길이
        overview = rec.get("overview", "") or ""
        objectives = rec.get("objectives", "") or ""
        weekly_plan = rec.get("weekly_plan", []) or []
        score = len(overview) + len(objectives) + len(weekly_plan)

        prev = best_by_key.get(key)
        if prev is None:
            best_by_key[key] = rec
        else:
            prev_overview = prev.get("overview", "") or ""
            prev_objectives = prev.get("objectives", "") or ""
            prev_weekly = prev.get("weekly_plan", []) or []
            prev_score = len(prev_overview) + len(prev_objectives) + len(prev_weekly)

            if score > prev_score:
                best_by_key[key] = rec

    return list(best_by_key.values())


def create_syllabus_text(doc: Dict[str, Any], include_header: bool = False) -> str:
    def safe_get(key, default=""):
        value = doc.get(key, default)
        if value is None:
            return default
        return str(value)

    parts: List[str] = []

    if include_header:
        suup_no = safe_get("suupNo")
        haksu_no = safe_get("haksuNo")
        name = safe_get("courseName")

        if suup_no:
            parts.append(f"수업번호: {suup_no}")
        if haksu_no:
            parts.append(f"학수번호: {haksu_no}")
        if name:
            parts.append(f"교과목명: {name}")

    overview = safe_get("overview")
    if overview:
        parts.append("과목 개요:")
        parts.append(overview)

    objectives = safe_get("objectives")
    if objectives:
        parts.append("학습 목표:")
        parts.append(objectives)

    weekly_plan = doc.get("weekly_plan", []) or []
    if weekly_plan:
        lines = ["주차별 수업 계획:"]
        for item in weekly_plan:
            week = item.get("week")
            topic = item.get("topic", "")
            if week is None and not topic:
                continue
            lines.append(f"{week}주차: {topic}")
        parts.append("\n".join(lines))

    year = safe_get("year")
    semester = safe_get("semester")
    if year or semester:
        parts.append(f"년도/학기: {year}년 {semester}")

    return "\n".join(parts)


def process_courses_with_syllabus(
    csv_path: str = "./data/raw/courses_2025_2.csv",
    syllabus_json_path: str = "./data/raw/syllabus_2025_2.json",
    chunk_size: int = 512,
    chunk_overlap: int = 102
) -> List[Dict[str, Any]]:
    
    # 1) syllabus 로드 및 중복 제거
    raw_syllabus = load_syllabus_json(syllabus_json_path)
    dedup_syllabus = deduplicate_syllabus(raw_syllabus)

    # 학수번호(haksuNo) → syllabus dict 매핑
    syllabus_by_course_id: Dict[str, Dict[str, Any]] = {}
    for rec in dedup_syllabus:
        haksu_no = str(rec.get("haksuNo", "")).strip()
        if haksu_no:
            syllabus_by_course_id[haksu_no] = rec

    # 2) courses CSV 로드
    df = pd.read_csv(csv_path, encoding="utf-8")

    # 수업번호 기준으로 그룹화 (한 클래스 단위)
    course_groups = df.groupby("학수번호")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )

    all_chunks: List[Dict[str, Any]] = []

    for course_id, group in course_groups:
        # 2-1) 과목 기본 정보 텍스트 (courses)
        course_texts: List[str] = []
        for _, row in group.iterrows():
            course_text = create_course_text(row)
            if course_text.strip():
                course_texts.append(course_text)

        if not course_texts:
            continue

        course_block = "\n\n".join(course_texts)

        # 2-2) syllabus 텍스트 (있으면)
        syllabus_rec = syllabus_by_course_id.get(str(course_id))
        syllabus_text = ""
        if syllabus_rec:
            syllabus_text = create_syllabus_text(syllabus_rec, include_header=True)
       
        # 2-3) 인덱싱에 사용하는 텍스트
        full_text = course_block

        # 2-4) 청킹
        if len(full_text) > chunk_size:
            chunks = text_splitter.split_text(full_text)
        else:
            chunks = [full_text]

        first_row = group.iloc[0]

        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "token_count": len(chunk.split()),
                "data_source": "courses",
                "class_no": str(first_row.get("수업번호", "")),
                "course_id": str(course_id),       
                "course_name": str(first_row.get("교과목명", "")),
                "professor": str(first_row.get("교강사", "")),
                "department": str(first_row.get("소속학과", "")),
                "semester": str(first_row.get("개설학기명", "")),
                "credit": str(first_row.get("학점", "")),
                "syllabus_text": syllabus_text,
                "original_text_length": len(full_text),
                "cleaned_text_length": len(chunk),
            })

    return all_chunks


#####################################################################

def merge_all_chunks(
    chunks_json: List[Dict[str, Any]],
    courses_syllabus_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """두 데이터 소스를 통합"""
    
    # 원래 chunks.json에 data_source가 있으면 그대로 두고,
    # 없으면 'haksa' 같은 기본값만 넣어주는 식으로
    for chunk in chunks_json:
        if 'data_source' not in chunk:
            chunk['data_source'] = "haksa"

    for chunk in courses_syllabus_chunks:
        if 'data_source' not in chunk:
            chunk['data_source'] = "courses"
    
    # 통합
    combined: List[Dict[str, Any]] = []
    combined.extend(chunks_json)
    combined.extend(courses_syllabus_chunks)
    
    # chunk_id 재정렬 + key 순서 정리
    all_chunks: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(combined):
        new_chunk: Dict[str, Any] = {"chunk_id": idx}
        # 기존 chunk_id가 있었다면 덮어쓴 셈이 됨
        for k, v in chunk.items():
            if k == "chunk_id":
                continue
            new_chunk[k] = v
        all_chunks.append(new_chunk)
    
    return all_chunks



def save_merged_chunks(chunks: List[Dict[str, Any]], output_path: str = "./data/processed/all_chunks.json"):
    """통합된 청크를 JSON 파일로 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 1. chunks.json 로드
    chunks_json = load_chunks_json("./data/processed/chunks.json")
    
    # 2. courses + syllabus 통합 처리 및 청킹
    course_syllabus_chunks = process_courses_with_syllabus(
        csv_path="./data/raw/courses_2025_2.csv",
        syllabus_json_path="./data/raw/syllabus_2025_2.json",
        chunk_size=512,
        chunk_overlap=102
    )

    # 3. 통합
    all_chunks = merge_all_chunks(chunks_json, course_syllabus_chunks)
    
    # 4. 저장
    save_merged_chunks(all_chunks, "./data/processed/all_chunks.json")