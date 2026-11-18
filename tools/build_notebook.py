#!/usr/bin/env python
"""Generate notebooks/colab_run.ipynb pinned to a given commit:  python tools/build_notebook.py <commit_sha>"""
import sys
from pathlib import Path

import nbformat as nbf

COMMIT = sys.argv[1]
REPO = "https://github.com/Saelee-Kittipop/Term-Project.git"
nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md(f"""
# การพยากรณ์ความผันผวนพอร์ต ETF ด้วย LSTM/Transformer และประโยชน์ของ VIX — Colab runner

Notebook นี้รัน repository ที่ **commit `{COMMIT}`** (ตรึงไว้เพื่อให้ทำซ้ำได้) ผลทุกอย่างถูกเขียนลง Google Drive ทันที
หาก runtime หลุด ให้รัน cell ตั้งค่า (ข้อ 0–3) ใหม่แล้วรัน cell ที่ค้างอยู่ซ้ำ ระบบจะทำต่อจากจุดเดิม

**ลำดับ:** 0 ตั้งค่า → 1 ทดสอบความถูกต้อง → 2 ดาวน์โหลดข้อมูลและ smoke (train/validation) → 3 formal: data + baselines →
4 train 48 runs → 5 freeze (ตรวจ validation) → 6 เปิด test ครั้งเดียว + backtest + กราฟ + handover → 7 mirror ไป Google Sheets → 8 สร้าง bundle ส่งกลับ

> ใช้ GPU: Runtime → Change runtime type → T4 GPU
""")
md("## 0. ตั้งค่า: mount Google Drive, ตรวจ GPU, clone repository ที่ commit ที่ตรึงไว้")
code(f"""
import os, subprocess, sys
from google.colab import drive
drive.mount('/content/drive')

GIT_COMMIT = "{COMMIT}"                      # do not change for the formal run
REPO_URL   = "{REPO}"
DRIVE_ROOT = "/content/drive/MyDrive/vix_term_project"   # data, checkpoints and all outputs
SMOKE_RUN  = "smoke_v1"
FORMAL_RUN = "formal_v1"
os.makedirs(DRIVE_ROOT, exist_ok=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"          # deterministic cuBLAS
os.environ["VOLVIX_GIT_COMMIT"] = GIT_COMMIT
!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "NO GPU - training will run on CPU"
""")
code("""
REPO_DIR = "/content/Term-Project"
if not os.path.exists(REPO_DIR):
    !git clone -q {REPO_URL} {REPO_DIR}
!git -C {REPO_DIR} fetch -q origin && git -C {REPO_DIR} checkout -q {GIT_COMMIT}
head = subprocess.check_output(["git", "-C", REPO_DIR, "rev-parse", "HEAD"]).decode().strip()
assert head == GIT_COMMIT, f"checked out {head}, expected {GIT_COMMIT}"
%cd {REPO_DIR}
print("repository pinned at", head)
""")
code("""
# torch is preinstalled in Colab (CUDA build) and is NOT reinstalled
!pip -q install "arch>=7.0" "yfinance>=0.2.50" "pandas_market_calendars>=4.4,<5" pyyaml pyarrow
import torch, pandas, numpy, arch, yfinance
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| pandas", pandas.__version__,
      "| numpy", numpy.__version__, "| arch", arch.__version__, "| yfinance", yfinance.__version__)
""")
md("## 1. ทดสอบความถูกต้องของโค้ด (ต้องผ่านทั้งหมดก่อนทำขั้นถัดไป)\nทดสอบ target alignment, purge, scaler, baselines, cutoff, การบัญชี backtest และ guard ของ test set ด้วยข้อมูลสังเคราะห์")
code("""
r = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True)
open(f"{DRIVE_ROOT}/pytest_{GIT_COMMIT[:8]}.log", "w").write(r.stdout + r.stderr)
print(r.stdout[-3000:], r.stderr[-2000:])
assert r.returncode == 0, "tests failed - stop here and send the log"
""")
md("""
## 2. ดาวน์โหลดข้อมูลจริง (raw snapshot ไม่ถูกแก้ไขภายหลัง) และ smoke check บน train/validation
Smoke ใช้ 1 config, 1 seed, 2 epochs เพื่อตรวจว่าทุกขั้นรันได้ **ผล smoke ไม่ใช่ผลการทดลอง** และ smoke ไม่สามารถเปิด test ได้
""")
code("""
!python scripts/run_pipeline.py --root {DRIVE_ROOT} --run-name {SMOKE_RUN} --override configs/smoke.yaml \\
    --snapshot latest --stages data,baselines,train,freeze,figures,report 2>&1 | tee {DRIVE_ROOT}/smoke.log
""")
code("""
import json, pandas as pd
from IPython.display import Markdown, display
snap_id = json.load(open(f"{DRIVE_ROOT}/runs/{SMOKE_RUN}/run_manifest.json"))["data"]["snapshot_id"]
print("snapshot used:", snap_id)
display(pd.read_csv(f"{DRIVE_ROOT}/runs/{SMOKE_RUN}/dataset/split_manifest.csv"))
display(pd.read_csv(f"{DRIVE_ROOT}/runs/{SMOKE_RUN}/dataset/alignment_checks.csv"))
display(Markdown(open(f"{DRIVE_ROOT}/runs/{SMOKE_RUN}/reports/data_quality_report_th.md").read()))
""")
md("## 3. Formal run: เตรียมข้อมูลจาก snapshot เดียวกับ smoke และคำนวณ baselines")
code("""
!python scripts/run_pipeline.py --root {DRIVE_ROOT} --run-name {FORMAL_RUN} --snapshot {snap_id} \\
    --stages data,baselines 2>&1 | tee -a {DRIVE_ROOT}/formal.log
print(json.dumps(json.load(open(f"{DRIVE_ROOT}/runs/{FORMAL_RUN}/baselines/baseline_info.json"))["garch"], indent=1)[:1500])
""")
md("""
## 4. ฝึก 48 runs (resumable)
หาก Colab หลุด ให้รันข้อ 0 ใหม่ แล้วรัน cell นี้ซ้ำ run ที่เสร็จแล้วจะถูกข้าม run ที่ค้างจะทำต่อจาก checkpoint ล่าสุด
""")
code("""
!python scripts/run_pipeline.py --root {DRIVE_ROOT} --run-name {FORMAL_RUN} --stages train 2>&1 | tee -a {DRIVE_ROOT}/formal.log
g = pd.read_csv(f"{DRIVE_ROOT}/runs/{FORMAL_RUN}/training/grid_summary.csv")
print(g["status"].value_counts())
""")
md("""
## 5. Freeze: เลือก config, คำนวณ τ, cutoffs และ static comparator จาก train/validation แล้วตรึงพร้อม hash
**หลังจากนี้ห้ามแก้ไข config, ข้อมูล, โมเดล หรือ cutoff** ตรวจตาราง validation ด้านล่างก่อนไปข้อ 6
""")
code("""
!python scripts/run_pipeline.py --root {DRIVE_ROOT} --run-name {FORMAL_RUN} --stages freeze 2>&1 | tee -a {DRIVE_ROOT}/formal.log
fz = json.load(open(f"{DRIVE_ROOT}/runs/{FORMAL_RUN}/frozen/freeze.json"))
print("tau =", fz["tau"]); print(json.dumps(fz["selection"], indent=1)); print(json.dumps(fz["cutoffs"], indent=1))
display(pd.read_csv(f"{DRIVE_ROOT}/runs/{FORMAL_RUN}/validation/validation_metrics.csv"))
""")
md("""
## 6. เปิด test ครั้งเดียว → backtest → กราฟ → เอกสาร handover
ระบบตรวจ hash ของทุกสิ่งที่ตรึงไว้ก่อนเปิด test และบันทึก `TEST_OPENED.json` การรันซ้ำต้องใช้ `--allow-test-rerun-reason` (เฉพาะแก้ข้อผิดพลาด implementation ไม่ใช่เพื่อปรับผล)
""")
code("""
!python scripts/run_pipeline.py --root {DRIVE_ROOT} --run-name {FORMAL_RUN} --stages test,backtest,figures,report 2>&1 | tee -a {DRIVE_ROOT}/formal.log
display(pd.read_csv(f"{DRIVE_ROOT}/runs/{FORMAL_RUN}/test/test_metrics.csv"))
display(Markdown(open(f"{DRIVE_ROOT}/runs/{FORMAL_RUN}/reports/handover_th.md").read()))
""")
md("## 7. Mirror ตารางสรุปไป Google Sheets (ไฟล์ CSV/JSON บน Drive ยังเป็นผลหลัก)")
code("""
from google.colab import auth
auth.authenticate_user()
!pip -q install gspread
SPREADSHEET_ID = "15CUiD2KzgJriNChW7yKtuZCbVqTN7tRq05nNRnP3O28"   # mirror sheet in MyDrive/vix_term_project; set "" to create a new one
args = f"--spreadsheet-id {SPREADSHEET_ID}" if SPREADSHEET_ID else ""
!python scripts/mirror_to_sheets.py --run-dir {DRIVE_ROOT}/runs/{FORMAL_RUN} {args}
""")
md("## 8. สร้าง bundle สำหรับส่งกลับ (ไม่รวม checkpoints/parquet ซึ่งเก็บอยู่ใน Drive)")
code("""
!python scripts/make_bundle.py --root {DRIVE_ROOT} --run-name {FORMAL_RUN} --extra-run {SMOKE_RUN}
from google.colab import files
files.download(f"{DRIVE_ROOT}/bundle_{FORMAL_RUN}.zip")
""")
nb["cells"] = C
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
out = Path(__file__).resolve().parents[1] / "notebooks" / "colab_run.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print(out)
