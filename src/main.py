from typing import Dict, Any, List
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.chat.chat_server import ChatSession


def run_cli():
    print("RAG + OpenAI 메인 파이프라인")
    session = ChatSession()

    while True:
        try:
            question = input("\n[질문]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n대화를 종료합니다.")
            break

        if not question:
            continue
        if question.lower() in ["exit", "quit", "q"]:
            print("대화를 종료합니다.")
            break

        # 이번 턴 RAG + LLM
        result: Dict[str, Any] = session.ask(
            question=question,
            k=5,
            model="gpt-4o-mini",
        )

        answer = result["answer"]

        print("\n[답변]")
        print(answer)


if __name__ == "__main__":
    run_cli()