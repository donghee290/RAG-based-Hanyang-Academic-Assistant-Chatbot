import re
import time
import requests
import pandas as pd
from typing import Any, Dict, List, Optional

PORTAL_BASE_URL = "https://portal.hanyang.ac.kr"
COURSE_LIST_PATH = "/sugang/SgscAct/findSuupSearchSugangSiganpyo.do"

class CoursesCrawler:
    def __init__(
        self,
        year: str,
        term: str,
        jojik_code: str,
        page_size: int = 20,
        delay_sec: float = 0.4,
    ):
        self.year = str(year)
        self.term = str(term)
        self.jojik_code = jojik_code
        self.page_size = page_size
        self.delay_sec = delay_sec

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                ),
                "Content-Type": "application/json+sua; charset=UTF-8",
            }
        )

    # ---------- 유틸 ----------

    @staticmethod
    def _extract_day(time_str: str) -> str:
        if not time_str:
            return ""
        m = re.match(r"^[가-힣]", time_str)
        return m.group(0) if m else ""

    @staticmethod
    def _time_to_period(hhmm: str) -> Optional[int]:
        try:
            h, m = map(int, hhmm.split(":"))
        except Exception:
            return None

        base_min = 8 * 60
        cur_min = h * 60 + m
        delta = cur_min - base_min

        if delta < 0 or delta >= 14 * 60:
            return None

        period = delta // 30 + 1
        return period


    @staticmethod
    def _extract_period(time_str: str) -> Dict[str, str]:
        if not time_str:
            return {"start": "", "end": ""}

        m = re.search(r"\((\d{2}:\d{2})-(\d{2}:\d{2})\)", time_str)
        if not m:
            return {"start": "", "end": ""}

        start_time = m.group(1)
        end_time = m.group(2)

        start_p = CoursesCrawler._time_to_period(start_time)

        try:
            eh, em = map(int, end_time.split(":"))
            end_total = eh * 60 + em - 30
            if end_total < 0:
                end_p = None
            else:
                end_h = end_total // 60
                end_m = end_total % 60
                end_base = f"{end_h:02d}:{end_m:02d}"
                end_p = CoursesCrawler._time_to_period(end_base)
        except Exception:
            end_p = None

        return {
            "start": str(start_p) if start_p is not None else "",
            "end": str(end_p) if end_p is not None else "",
        }

    @staticmethod
    def _extract_time_range(time_str: str) -> Dict[str, str]:
        if not time_str:
            return {"start_time": "", "end_time": ""}
        m = re.search(r"\((\d{2}:\d{2})-(\d{2}:\d{2})\)", time_str)
        if not m:
            return {"start_time": "", "end_time": ""}
        return {"start_time": m.group(1), "end_time": m.group(2)}

    @staticmethod
    def _extract_page_list(raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(raw_json, dict) or not raw_json:
            return []

        key = next(iter(raw_json.keys()), None)
        if not key:
            return []

        try:
            value = raw_json[key]
            if not isinstance(value, list) or not value:
                return []
            first = value[0]
            data_list = first.get("list")
            if not data_list:
                return []
            return data_list
        except Exception:
            return []

    # ---------- 페이지 단위 수강편람 요청 ----------

    def _build_payload(self, page_index: int) -> Dict[str, Any]:
        skip_rows = page_index * self.page_size

        return {
            "skipRows": str(skip_rows),
            "maxRows": str(self.page_size),
            "notAppendQrys": "true",
            "strLocaleGb": "ko",
            "strIsSugangSys": "true",
            "strDetailGb": "0",
            "strDaehak": "",
            "strGwamok": "",
            "strHakgwa": "",
            "strSuupNo": "",
            "strHaksuNo": "",
            "strIlbanCommonGb": "",
            "strIsuGbCd": "",
            "strIsuGrade": "",
            "strJojik": self.jojik_code,
            "strSuupOprGb": "0",
            "strSuupTerm": self.term,
            "strSuupYear": self.year,
            "strTsGangjwa": "",
            "strTsGangjwa3": "0",
            "strTsGangjwaAll": "0",
            "strYeongyeok": "",
        }

    def fetch_page(self, page_index: int) -> List[Dict[str, Any]]:
        url = PORTAL_BASE_URL + COURSE_LIST_PATH
        payload = self._build_payload(page_index)

        resp = self.session.post(url, json=payload)
        resp.raise_for_status()

        raw_json = resp.json()
        data_list = self._extract_page_list(raw_json)
        return data_list

    # ---------- 과목 1개 평탄화 ----------

    def _flatten_course(self, course_raw: Dict[str, Any]) -> List[Dict[str, Any]]:
        times_str = course_raw.get("suupTimes") or ""
        rooms_str = course_raw.get("suupRoomNms") or ""

        time_tokens = re.findall(r"[가-힣]\(\d{2}:\d{2}-\d{2}:\d{2}\)", times_str)
        if not time_tokens:
            time_tokens = [None]
        room_tokens = [s.strip() for s in rooms_str.split(",")] if rooms_str else []

        flattened_rows: List[Dict[str, Any]] = []

        for idx, time_str in enumerate(time_tokens):
            row = {
                "수업번호": course_raw.get("suupNo", ""),
                "학수번호": course_raw.get("haksuNo", ""),
                "교과목명": course_raw.get("gwamokNm", ""),
                "영문명": course_raw.get("gwamokEnm", ""),
                "학점": course_raw.get("hakjeom", ""),
                "이수구분": course_raw.get("isuGbNm", ""),
                "이수단위": course_raw.get("isuUnitNm", ""),
                "영역": course_raw.get("yungyukNm", ""),
                "학년": course_raw.get("banGrade", ""),
                "개설학기명": course_raw.get("isuTermNm", ""),
                "교강사": course_raw.get("gyogangsaNms", ""),
                "강좌유형": course_raw.get("suupTypeNm", ""),
                "정원": course_raw.get("jehanInwon", ""),
                "소속학과": course_raw.get("slgSosokNm", ""),
                "관장학과": course_raw.get("gnjSosokNm", ""),
                #"수업시간(raw)": course_raw.get("suupTimes", ""),
                #"강의실(raw)": course_raw.get("suupRoomNms", ""),
                "6C핵심역량": course_raw.get("yrGbNm", "")
            }

            if time_str:
                day = self._extract_day(time_str)
                period = self._extract_period(time_str)
                time_range = self._extract_time_range(time_str)

                room = room_tokens[idx] if idx < len(room_tokens) else ""

                if room.startswith("서울 "):
                    room = room.replace("서울 ", "", 1)

                row.update(
                    {
                        "요일": day,
                        #"시간표": time_str,
                        "시작교시": period["start"],
                        "종료교시": period["end"],
                        "시작시간": time_range["start_time"],
                        "종료시간": time_range["end_time"],
                        "강의실": room,
                    }
                )
            else:
                row.update(
                    {
                        "요일": "",
                        #"시간표": "",
                        "시작교시": "",
                        "종료교시": "",
                        "시작시간": "",
                        "종료시간": "",
                        "강의실": "",
                    }
                )

            flattened_rows.append(row)

        return flattened_rows

    # ---------- 전체 크롤링 ----------

    def crawl_all(self, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
        all_rows: List[Dict[str, Any]] = []
        page_index = 0

        while True:
            if (max_pages is not None) and (page_index >= max_pages):
                print(f"[INFO] 페이지 {max_pages}까지만 크롤링하고 중단합니다.")
                break

            try:
                page_data = self.fetch_page(page_index)
            except Exception as e:
                print(f"[ERROR] 페이지 {page_index + 1} 요청 실패: {e}")
                break

            if not page_data:
                print("[INFO] 더 이상 데이터가 없습니다. 종료합니다.")
                break

            for course_raw in page_data:
                flattened = self._flatten_course(course_raw)
                all_rows.extend(flattened)

            if len(page_data) < self.page_size:
                print("[INFO] 마지막 페이지로 판단됩니다. 종료합니다.")
                break

            page_index += 1
            time.sleep(self.delay_sec)

        print(f"[DONE] 총 {len(all_rows)}행(시간 슬롯 기준) 수집 완료")
        return all_rows

    # ---------- CSV 저장 ----------

    @staticmethod
    def save_to_csv(rows: List[Dict[str, Any]], filepath: str):
        if not rows:
            print("저장할 데이터가 없습니다.")
            return

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        print(f"[SAVE] CSV 저장 완료 → {filepath}")

def main():
    crawler = CoursesCrawler( year="2025", term="20", jojik_code="H0002256", page_size=20, delay_sec=0.4 )
    # rows = crawler.crawl_all(max_pages=1)
    rows = crawler.crawl_all()
    crawler.save_to_csv(rows, "./data/raw/courses_2025_2.csv")
if __name__ == "__main__":
    main()