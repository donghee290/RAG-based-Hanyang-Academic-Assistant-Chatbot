import time
import json
import csv
import os
from playwright.sync_api import sync_playwright

class SyllabusCrawler:
    def __init__(self, year, semester):
        self.year = year
        self.semester = semester
        self.base_popup_url = "https://portal.hanyang.ac.kr/openPop.do"
        self.results = []
        self.log_file = "log.txt"
        
        # Clear log file on start
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("Log started\n")

    def log_message(self, message):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")

    def is_generic_topic(self, topic):
        generic_phrases = [
            "to be announced",
            "find the attached file",
            "syllabus",
            "강의계획서",
            "참조",
            "추후공지",
            "비어있음"
        ]
        if not topic:
            return True
        
        topic_lower = topic.lower()
        for phrase in generic_phrases:
            if phrase in topic_lower:
                return True
        return False

    def crawl_from_csv(self, csv_path, output_file):
        print(f"Reading courses from {csv_path}...")
        courses_to_crawl = []
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                header = next(reader) # Skip header
                # Assuming index 0 is suupNo, index 1 is haksuNo based on inspection
                for row in reader:
                    if len(row) >= 2:
                        courses_to_crawl.append({'suupNo': row[0], 'haksuNo': row[1]})
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return

        print(f"Found {len(courses_to_crawl)} courses to crawl.")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            count = 0
            
            for course in courses_to_crawl:
                suup_no = course['suupNo']
                expected_haksu_no = course['haksuNo']
                
                url = f"{self.base_popup_url}?header=hidden&url=/haksa/SughAct/findSuupPlanDocHyIn.do&flag=DN&year={self.year}&term={self.semester}&suup={suup_no}&language=ko"
                
                try:
                    page.goto(url, timeout=5000)
                    
                    try:
                        page.wait_for_selector("text='교과목명'", timeout=3000)
                    except:
                        pass
                    
                    # Extract data using JS
                    data_extracted = page.evaluate("""() => {
                        const getText = (id) => {
                            const el = document.getElementById(id);
                            return el ? el.innerText.trim() : "";
                        };
                        const getValue = (id) => {
                            const el = document.getElementById(id);
                            return el ? el.value.trim() : "";
                        };

                        const haksuNo = getText('haksuNo');
                        const courseName = getText('gwamokNm');
                        const overview = getValue('gwamokYoyak');
                        const objectives = getValue('gwamokMokpyo');

                        // Weekly Plan
                        const weekly_plan = [];
                        const weekRows = document.querySelectorAll('#gdWeek tbody tr');
                        weekRows.forEach(row => {
                            const weekElem = row.querySelector('#jucha');
                            const subjectElem = row.querySelector('input[name="subject"]');
                            
                            if (weekElem && subjectElem) {
                                const week = parseInt(weekElem.innerText.trim());
                                const subject = subjectElem.value.trim();
                                
                                if (!isNaN(week)) {
                                    weekly_plan.push({ week: week, topic: subject });
                                }
                            }
                        });

                        return {
                            haksuNo, courseName, overview, objectives, weekly_plan
                        };
                    }""")
                    
                    # Validation
                    if data_extracted['haksuNo'] != expected_haksu_no:
                        print(f"[{suup_no}] Mismatch: Expected {expected_haksu_no}, got {data_extracted['haksuNo']}")
                        # We still save it, but maybe log it? For now just print.
                    
                    # Data Cleaning for Weekly Plan
                    weekly_plan = data_extracted['weekly_plan']
                    is_all_generic = True
                    if not weekly_plan:
                        is_all_generic = True # Empty is treated as generic/empty
                    else:
                        for item in weekly_plan:
                            if not self.is_generic_topic(item['topic']):
                                is_all_generic = False
                                break
                    
                    if is_all_generic:
                        self.log_message(f"[{suup_no}] Generic weekly plan detected. Cleared.")
                        weekly_plan = []
                    
                    course_data = {
                        "suupNo": suup_no,
                        "haksuNo": data_extracted['haksuNo'],
                        "courseName": data_extracted['courseName'],
                        "overview": data_extracted['overview'],
                        "objectives": data_extracted['objectives'],
                        "weekly_plan": weekly_plan,
                        "year": self.year,
                        "semester": self.semester
                    }
                    
                    self.results.append(course_data)
                    print(f"[{suup_no}] Processed: {data_extracted['courseName']}")
                    
                except Exception as e:
                    print(f"[{suup_no}] Error: {e}")
                    self.log_message(f"[{suup_no}] Error: {e}")
                
                count += 1
                
                # Periodic save
                if count % 50 == 0:
                    print(f"Saving progress... ({count}/{len(courses_to_crawl)})")
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(self.results, f, ensure_ascii=False, indent=4)
            
            browser.close()
            
        # Final save
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=4)
        print(f"Finished. Saved {len(self.results)} courses to {output_file}")

def main():
    crawler = SyllabusCrawler("2025", "20")
    csv_path = "./data/raw/courses_2025_2.csv"
    output_file = "./data/raw/syllabus_2025_2.json"
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        # Try full path if relative fails, though cwd should be correct
        csv_path = r"d:\Users\donghee\TextMining\courses_2025_2.csv"
    
    crawler.crawl_from_csv(csv_path, output_file)

if __name__ == "__main__":
    main()
