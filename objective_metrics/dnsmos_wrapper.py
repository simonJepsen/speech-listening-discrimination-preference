#!/usr/bin/env python3
import argparse, importlib.util, json, os
from pathlib import Path

def load_module(path):
    spec = importlib.util.spec_from_file_location("official_dnsmos_local", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

p = argparse.ArgumentParser()
p.add_argument("wav", type=Path)
p.add_argument("--repo", type=Path,
               default=Path(os.environ["DNSMOS_REPO"]) if "DNSMOS_REPO" in os.environ else None)
p.add_argument("--personalized", action="store_true")
a = p.parse_args()

if a.repo is None:
    raise SystemExit("Set DNSMOS_REPO or pass --repo /path/to/DNS-Challenge")

repo = a.repo.resolve()
script = repo / "DNSMOS" / "dnsmos_local.py"

p808 = (
    repo
    / "DNSMOS"
    / "DNSMOS"
    / "model_v8.onnx"
)

if a.personalized:
    primary = (
        repo
        / "DNSMOS"
        / "pDNSMOS"
        / "sig_bak_ovr.onnx"
    )
else:
    primary = (
        repo
        / "DNSMOS"
        / "DNSMOS"
        / "sig_bak_ovr.onnx"
    )
wav = a.wav.resolve()

for q in (script, p808, primary, wav):
    if not q.exists(): raise FileNotFoundError(q)

os.chdir(repo)
m = load_module(script)
scorer = m.ComputeScore(str(primary), str(p808))
r = scorer(str(wav), m.SAMPLING_RATE, a.personalized)
print(json.dumps({
    "sig": float(r["SIG"]),
    "bak": float(r["BAK"]),
    "ovrl": float(r["OVRL"]),
    "p808_mos": float(r["P808_MOS"]),
}))
