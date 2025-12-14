# -*- coding: utf-8 -*-
"""
청킹 및 전처리 모듈
- 네비게이션 메뉴 및 푸터 제거
- 텍스트 청킹 (512 토큰, 20% 오버랩)
"""

import os
import json
import re
from typing import List, Dict, Any
from pathlib import Path

# tiktoken을 사용하여 정확한 토큰 계산 (OpenAI와 동일)
try:
    import tiktoken
except ImportError:
    print("tiktoken 설치 필요: pip install tiktoken")
    tiktoken = None


# 전처리 함수들
def remove_navigation_menu(text: str) -> str:
    """상단 네비게이션 메뉴 제거"""
    lines = text.split('\n')
    cleaned_lines = []
    skip_patterns = [
        r'^학사 안내$',
        r'^캠퍼스 소개$',
        r'^행정 지원 안내$',
        r'^기관/단체$',
        r'^대학 생활$',
        r'^학사일정$',
        r'^수강신청$',
        r'^초기화면 바로가기$',
        r'^서울캠퍼스$',
        r'^한양소개$',
        r'^ERICA캠퍼스$',
        r'^입학/교육$',
        r'^탐색$',
        r'^내용으로 건너뛰기$',
        r'^연구/산학/창업$',
        r'^수업 안내$',
        r'^온라인 강좌$',
        r'^전공제도$',
        r'^학적변동$',
        r'^성적/졸업$',
        r'^학점인정$',
        r'^계절학기$',
        r'^학점 교류$',
        r'^서울 학사안내$',
        r'^학적$',
        r'^학적변동 신청$',
        r'^학적사항 정정$',
        r'^휴학$',
        r'^복학$',
        r'^자진유급$',
        r'^자퇴/제적$',
        r'^재입학$',
        r'^전과$',
        r'^FAQ$',
        r'^전공$',
        r'^부전공$',
        r'^다중/융합전공$',
        r'^복수전공$',
        r'^마이크로전공$',
        r'^교육과정$',
        r'^2024-2027 교육과정$',
        r'^2020-2023 교육과정$',
        r'^2016-2019 교육과정안내$',
        r'^2013-2015 교육과정안내$',
        r'^2009-2012 교육과정안내$',
        r'^수업$',
        r'^정규학기 수업안내$',
        r'^계절학기 수업안내$',
        r'^학점교류$',
        r'^강의동 및 행정팀안내$',
        r'^성적$',
        r'^성적$',
        r'^학점포기$',
        r'^외국대학취득학점인정$',
        r'^인턴십$',
        r'^졸업$',
        r'^졸업요건$',
        r'^조기졸업$',
        r'^학사학위취득유예$',
        r'^학적변동 신청/문의$',
        r'^증명$',
        r'^증명발급 안내$',
        r'^증명발급 방법$',
        r'^원본대조확인$',
        r'^자료실$',
        r'^수업자료실$',
        r'^학적자료실$',
        r'^공지사항$',
        r'^업무양식다운$',
        r'^국내대학간 학점교류안내$',
        r'^졸업요건 - 서울 학사안내$',
        r'^강의평가$',
        r'^강의시간표$',
        r'^온라인강좌안내$',
        r'^특수수업$',
        r'^스마트출결시스템$',
        r'^수업도우미$',
        r'^계절학기 안내$',
        r'^계절학기 수강신청$',
        r'^계절학기 수요조사$',
        r'^부분환불원칙$',
        r'^개설예정 전공과목$',
        r'^본교생 학점교류\(out\)$',
        r'^타대학생 학점교류\(in\)$',
        r'^서울캠퍼스는$',
        r'^대학/학과 소개$',
        r'^대학원$',
        r'^캠퍼스 지도\+찾아오는길$',
        r'^주차안내$',
        r'^한양둘레길\(8경\)$',
        r'^대학탐방 신청$',
        r'^캠퍼스 매거진$',
        r'^기숙사 안내$',
        r'^업무별 담당 부서 안내$',
        r'^등록 안내$',
        r'^장학 안내$',
        r'^교직이수$',
        r'^교원자격증발급$',
        r'^학생대관$',
        r'^외부대관$',
        r'^통합 양식자료실$',
        r'^행정기관$',
        r'^부속/부설기관$',
        r'^학생 기구$',
        r'^동아리 안내$',
        r'^학생 언론사$',
        r'^병무안내$',
        r'^고시반 안내$',
        r'^2024 캠퍼스 가이드북$',
        r'^오늘의 메뉴$',
        r'^IT 서비스$',
        r'^복지 매장$',
        r'^복지 혜택$',
        r'^보건 안내$',
        r'^의료혜택$',
        r'^학생 상담$',
        r'^인권센터$',
        r'^학생증/국제학생증$',
        r'^입학총괄$',
        r'^인재개발$',
        r'^학술 정보$',
        r'^글로벌 교육$',
        r'^평생교육$',
        r'^사회봉사$',
        r'^사랑의 실천$',
        r'^주전공 이수 조건$',
        r'^비전 및 목표$',
        r'^이수단위 정의$',
        r'^리더십 인증 로드맵$',
        r'^인터넷증명발급\(HY-in\)$',
        r'^FAX민원\(정부24\)$',
        r'^우편증명발급$',
        r'^학교 방문발급$',
        r'^학적부 정정\(성명/주민등록번호/국적\)$',
        r'^신상정보 정정\(영문성명/연락처/주소\)$',
        r'^재입학 안내$',
        r'^재입학운영내규$',
        r'^국문\(KOR\)$',
        r'^영문\(ENG\)$',
        r'^인문과학대학$',
        r'^공과대학$',
        r'^사회과학대학$',
        r'^경제금융대학$',
        r'^경영대학$',
        r'^생활과학대학$',
        r'^자연과학대학$',
        r'^국제대학$',
        r'^융합전공대학$',
        r'^미래인문학교육인증센터$',
        r'^Micro Major$',
        r'^주전공 이수조건$',
        r'^주전공이수조건$',
        r'^이수구분 정의$',
        r'^개편철학 및 전략$',
        r'^과목구분 정의$',
        r'^제2전공 이수조건$',
        r'^교양교육과정 운영$',
        r'^과목구분 처리기준$',
        r'^평가방법$',
        r'^학과\(부\) 교육과정 편성절차$',
        r'^개편된 주요내용$',
        r'^달라진 학사제도$',
    ]
    
    # 반복되는 메뉴 패턴 제거 (연속으로 3개 이상 나오는 경우)
    menu_count = 0
    for i, line in enumerate(lines):
        is_menu = any(re.match(pattern, line.strip()) for pattern in skip_patterns)
        
        if is_menu:
            menu_count += 1
            if menu_count > 2:  # 연속으로 메뉴가 많이 나오면 스킵
                continue
        else:
            menu_count = 0
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def remove_footer(text: str) -> str:
    """하단 푸터 정보 제거"""
    # 푸터 시작 패턴들
    footer_patterns = [
        r'콘텐츠 담당부서',
        r'COPYRIGHT',
        r'대학 홈페이지',
        r'대학원 홈페이지',
        r'서울 캠퍼스',
        r'ERICA 캠퍼스',
        r'TEL\.',
        r'facebook',
        r'youtube',
        r'=== 사이트로 이동 ===',
        r'숨김',
    ]
    
    lines = text.split('\n')
    cleaned_lines = []
    footer_started = False
    
    for line in lines:
        # 푸터 시작 감지
        if any(re.search(pattern, line) for pattern in footer_patterns):
            footer_started = True
        
        if not footer_started:
            cleaned_lines.append(line)
        else:
            # 푸터 이후에 실제 콘텐츠가 다시 나올 수 있으므로
            # 특정 패턴이 나오면 푸터 종료로 간주
            if re.match(r'^\d+\.', line.strip()):  # 번호 목록으로 시작
                footer_started = False
                cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def clean_text(text: str) -> str:
    """전체 전처리 파이프라인"""
    # 1. 네비게이션 메뉴 제거
    text = remove_navigation_menu(text)
    
    # 2. 푸터 제거
    text = remove_footer(text)
    
    # 3. 빈 줄 정리
    lines = [line.strip() for line in text.split('\n')]
    lines = [line for line in lines if line]  # 빈 줄 제거
    
    # 4. 과도한 공백 제거
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)  # 3개 이상 연속 줄바꿈을 2개로
    
    return text.strip()


# 토큰 계산 함수
def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """텍스트의 토큰 수 계산"""
    if tiktoken is None:
        # 대략적인 추정: 한글 1자 = 1.5 토큰, 영문/숫자 1자 = 0.25 토큰
        korean_chars = len(re.findall(r'[가-힣]', text))
        other_chars = len(text) - korean_chars
        return int(korean_chars * 1.5 + other_chars * 0.25)
    
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


# 청킹 함수
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 102) -> List[Dict[str, Any]]:
    """
    텍스트를 청크로 분할
    
    Args:
        text: 분할할 텍스트
        chunk_size: 청크 크기 (토큰 수)
        overlap: 오버랩 크기 (토큰 수, 20% = 512 * 0.2 = 102)
    
    Returns:
        청크 리스트 (각 청크는 텍스트와 메타데이터 포함)
    """
    chunks = []
    
    # 문단 단위로 분할 (더 자연스러운 청킹)
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    current_tokens = 0
    chunk_idx = 0
    
    for para in paragraphs:
        para_tokens = count_tokens(para)
        
        # 현재 청크에 추가했을 때 크기 초과 여부 확인
        if current_tokens + para_tokens > chunk_size and current_chunk:
            # 현재 청크 저장
            chunks.append({
                "chunk_id": chunk_idx,
                "text": current_chunk.strip(),
                "token_count": current_tokens,
                "start_char": len('\n\n'.join(chunks)) if chunks else 0,
            })
            
            # 오버랩을 위해 이전 청크의 마지막 부분 가져오기
            if overlap > 0:
                # 이전 청크의 마지막 부분을 오버랩 크기만큼 가져오기
                prev_text = current_chunk
                prev_tokens = count_tokens(prev_text)
                
                # 오버랩 크기만큼 텍스트 추출 (뒤에서부터)
                overlap_text = ""
                overlap_tokens = 0
                words = prev_text.split()
                for word in reversed(words):
                    word_tokens = count_tokens(word + " ")
                    if overlap_tokens + word_tokens <= overlap:
                        overlap_text = word + " " + overlap_text
                        overlap_tokens += word_tokens
                    else:
                        break
                
                current_chunk = overlap_text + para
                current_tokens = overlap_tokens + para_tokens
            else:
                current_chunk = para
                current_tokens = para_tokens
            
            chunk_idx += 1
        else:
            # 현재 청크에 추가
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
            current_tokens += para_tokens
    
    # 마지막 청크 추가
    if current_chunk:
        chunks.append({
            "chunk_id": chunk_idx,
            "text": current_chunk.strip(),
            "token_count": current_tokens,
        })
    
    return chunks


def process_crawled_data(
    base_dir: str = "./data/raw/hyu_pages",
    selected_titles: List[str] = None
) -> List[Dict[str, Any]]:
    """
    크롤링된 데이터를 전처리하고 청킹
    
    Args:
        base_dir: 크롤링된 데이터가 저장된 디렉토리
        selected_titles: 처리할 페이지 제목 리스트 (None이면 모두 처리)
    
    Returns:
        청크 리스트 (각 청크에 원본 메타데이터 포함)
    """
    all_chunks = []
    base_path = Path(base_dir)
    
    # 모든 페이지 디렉토리 찾기
    for page_dir in base_path.iterdir():
        if not page_dir.is_dir():
            continue
        
        meta_path = page_dir / "meta.json"
        content_path = page_dir / "content.txt"
        
        if not (meta_path.exists() and content_path.exists()):
            continue
        
        # 메타데이터 읽기
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        # 제목 필터링
        if selected_titles and meta.get('title') not in selected_titles:
            continue
        
        # 콘텐츠 읽기
        with open(content_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        # 전처리
        cleaned_text = clean_text(raw_text)
        
        # 청킹
        chunks = chunk_text(cleaned_text, chunk_size=512, overlap=102)
        
        # 각 청크에 원본 메타데이터 추가
        for chunk in chunks:
            chunk.update({
                "source_category": meta.get('category'),
                "source_title": meta.get('title'),
                "source_url": meta.get('url'),
                "original_text_length": len(raw_text),
                "cleaned_text_length": len(cleaned_text),
            })
        
        all_chunks.extend(chunks)
        
        print(f"[PROCESSED] {meta.get('title')}: {len(chunks)} chunks")
    
    return all_chunks


def save_chunks(chunks: List[Dict[str, Any]], output_path: str = "./data/processed/chunks.json"):
    """청크를 JSON 파일로 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {len(chunks)} chunks -> {output_path}")


if __name__ == "__main__":
    # 모든 페이지 청킹 (selected_titles=None이면 모든 페이지 처리)
    # 특정 페이지만 선택하려면 selected 리스트를 사용
    # selected = ["학사일정", "졸업요건", "수강신청"]  # 예시
    
    # 데이터 처리 (모든 페이지)
    chunks = process_crawled_data(
        base_dir="./data/raw/hyu_pages",
        selected_titles=None  # None이면 모든 페이지 처리
    )
    
    # 저장
    save_chunks(chunks, "./data/processed/chunks.json")
    
    print(f"\n[SUMMARY]")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average tokens per chunk: {sum(c['token_count'] for c in chunks) / len(chunks):.1f}")