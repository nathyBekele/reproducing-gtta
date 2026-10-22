"""Assemble the Hugging Face Space folder from the repo's sources.

    python scripts/export_explorer.py && python scripts/export_live_tasks.py && python scripts/build_space.py

space/ then holds everything the Space needs: app.py, sa_live.py, data/tasks.json, static/ (pages, data, report).
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "space"
STATIC = SPACE / "static"
STATIC.mkdir(parents=True, exist_ok=True)

copies = {
    ROOT / "explorer" / "index.html": STATIC / "index.html",
    ROOT / "explorer" / "try.html": STATIC / "try.html",
    ROOT / "explorer" / "404.html": STATIC / "404.html",
    ROOT / "explorer" / "dg.html": STATIC / "dg.html",
    ROOT / "explorer" / "dg_data.js": STATIC / "dg_data.js",
    ROOT / "explorer" / "data.js": STATIC / "data.js",
    ROOT / "explorer" / "site.js": STATIC / "site.js",
    ROOT / "explorer" / "site.css": STATIC / "site.css",
    ROOT / "explorer" / "favicon.svg": STATIC / "favicon.svg",
    ROOT / "explorer" / "tracker.js": STATIC / "tracker.js",
    ROOT / "report" / "main.pdf": STATIC / "report.pdf",
    ROOT / "sa" / "live.py": SPACE / "sa_live.py",
}
for src, dst in copies.items():
    shutil.copyfile(src, dst)
    print(f"{src.relative_to(ROOT)} → {dst.relative_to(ROOT)}")
# Cache busting: tag each asset reference with a short content hash, so a returning visitor's browser
# never pairs a new page with an old cached site.js / site.css / data file.
import hashlib
import re

def _v(name):
    return hashlib.md5((STATIC / name).read_bytes()).hexdigest()[:10]

js = (STATIC / "site.js").read_text()      # site.js loads tracker.js itself
(STATIC / "site.js").write_text(js.replace('s.src = "tracker.js"', f's.src = "tracker.js?v={_v("tracker.js")}"'))
ASSETS = ["site.js", "site.css", "tracker.js", "dg_data.js", "data.js", "favicon.svg"]
for page in ["index.html", "try.html", "dg.html", "404.html"]:
    f = STATIC / page
    html = f.read_text()
    for name in ASSETS:
        html = re.sub(r'(?<![\w.-])' + re.escape(name) + r'(?=["\'])', f"{name}?v={_v(name)}", html)
    f.write_text(html)
print("asset versions added")
assert (SPACE / "data" / "tasks.json").exists(), "run scripts/export_live_tasks.py first"
assert not (STATIC / "config.js").exists(), "config.js is served by app.py; don't ship a static one"
print("space/ ready")
