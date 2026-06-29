import json, time, requests
from pathlib import Path

RAW = Path("data/raw")
MARKER = "この記事の続きを読むには"
UA = {"User-Agent": "Mozilla/5.0 (research; personal RAG project)"}

gated = []
for f in sorted(RAW.glob("*.json")):
    entries = json.loads(f.read_text())
    urls = {e["source_url"] for e in entries if e.get("source_url")}
    truncated = False
    for u in urls:
        r = requests.get(u, headers=UA, timeout=20)
        if MARKER in r.text:
            truncated = True
            break
        time.sleep(3)
    print(f"{'GATED ' if truncated else 'ok    '} {f.name} ({len(entries)})")
    if truncated:
        gated.append(f.name)
    time.sleep(3)

print("\n=== gated ===\n" + "\n".join(gated))