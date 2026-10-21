"""BFCLv3 tasks for the live demo's picker: full prompt + gold first call (GTTA split, commit 24ba304).

    python scripts/export_live_tasks.py     → space/data/tasks.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sa import data  # noqa: E402

out = []
for split in ("val", "test"):
    for ex in data.load_split(split):
        out.append({"id": ex.id, "split": split, "classes": ex.involved_classes, "request": ex.user_request,
                    "prompt": ex.prompt, "answer": ex.answer})
out.sort(key=lambda t: int(t["id"].rsplit("_", 1)[1]))
dest = ROOT / "space" / "data" / "tasks.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, ensure_ascii=False))
print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB, {len(out)} tasks)")
