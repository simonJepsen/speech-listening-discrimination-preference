#!/usr/bin/env python3
import argparse, csv, json, os, subprocess, sys, tempfile
from pathlib import Path

ALIASES = {
    "mos": ["mos_pred", "mos", "MOS"],
    "noisiness": ["noi_pred", "noisiness", "Noisiness"],
    "coloration": ["col_pred", "coloration", "Coloration"],
    "discontinuity": ["dis_pred", "discontinuity", "Discontinuity"],
    "loudness": ["loud_pred", "loudness", "Loudness"],
}
def val(row, names):
    for k in names:
        if k in row and row[k] not in ("", None):
            return float(row[k])
    raise KeyError(f"Missing {names}; columns={list(row)}")

p = argparse.ArgumentParser()
p.add_argument("wav", type=Path)
p.add_argument("--repo", type=Path,
               default=Path(os.environ["NISQA_REPO"]) if "NISQA_REPO" in os.environ else None)
p.add_argument("--model", type=Path,
               default=Path(os.environ["NISQA_MODEL"]) if "NISQA_MODEL" in os.environ else None)
p.add_argument("--python", default=os.environ.get("NISQA_PYTHON", sys.executable))
a = p.parse_args()

if a.repo is None:
    raise SystemExit("Set NISQA_REPO or pass --repo /path/to/NISQA")
repo = a.repo.resolve()
model = (a.model or (repo/"weights"/"nisqa.tar")).resolve()
wav = a.wav.resolve()
run_predict = repo/"run_predict.py"

with tempfile.TemporaryDirectory(prefix="nisqa_") as td:
    out = Path(td)
    cmd = [a.python, str(run_predict), "--mode", "predict_file",
           "--pretrained_model", str(model), "--deg", str(wav),
           "--output_dir", str(out), "--num_workers", "0", "--bs", "1"]
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + "\n" + proc.stderr)
    csvs = sorted(out.rglob("*.csv"), key=lambda q: q.stat().st_mtime, reverse=True)
    if not csvs: raise RuntimeError("NISQA produced no CSV")
    with csvs[0].open("r", encoding="utf-8-sig", newline="") as f:
        row = next(csv.DictReader(f), None)
    if row is None: raise RuntimeError("Empty NISQA CSV")
    print(json.dumps({k: val(row, v) for k, v in ALIASES.items()}))
