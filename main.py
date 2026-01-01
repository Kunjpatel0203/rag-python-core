from dotenv import load_dotenv
load_dotenv()

from rag.pipeline import ingest_blocks, query_blocks
from structured.router import route_question
from phase2_runner import run_phase2


def get_blocks():
    blocks = []
    for i in range(1, 6):
        print(f"\nBlock {i}")
        t = input("Type (url/pdf/doc/xls/ppt/skip): ").lower()

        if t == "skip":
            blocks.append({"block_id": i, "type": None, "value": None})
            continue

        value = input("Enter URL or file path: ").strip()
        blocks.append({
            "block_id": i,
            "type": t.upper(),
            "value": value
        })
    return blocks


blocks = get_blocks()
store = ingest_blocks(blocks, compliance_id="c1")

while True:
    q = input("\nAsk a question (or exit): ")
    if q.lower() == "exit":
        break
    ans = run_phase2(q, compliance_id="c1", rag_store=store)
    print("\nANSWER:\n", ans)
    

