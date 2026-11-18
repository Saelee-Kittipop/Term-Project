# การพยากรณ์ความผันผวนพอร์ต ETF ด้วย LSTM และ Transformer: ประโยชน์ของข้อมูล VIX

Repository สำหรับ term project วิชา Applied Deep Learning and Neural Networks ดำเนินการตามร่างรายงาน
`portfolio_volatility_research_th_2008_core60.docx` (ฉบับปรับปรุง 2) ซึ่งเป็น specification หลัก

Notebook ตรึง commit โค้ด `b9aaeb1cf45b6d0a94d83bcb96415d70a065b08d` ไว้ภายใน

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Saelee-Kittipop/Term-Project/blob/research-pipeline/notebooks/colab_run.ipynb)

## สถานะ
- โค้ดครบทุกขั้นตอน: ข้อมูล → target/split/scaler → HV20/EWMA/GARCH → LSTM/Transformer 48 runs → freeze → test → backtest → กราฟ → handover
- ชุดทดสอบความถูกต้อง (`tests/`) ผ่านทั้งหมด และได้รันทุกขั้นแบบครบวงจรด้วย **ข้อมูลสังเคราะห์** เพื่อตรวจ code path แล้ว
- **ยังไม่มีผลการทดลองจากข้อมูลจริง** ต้องรันบน Colab (ข้อมูล Yahoo/Cboe/French ดาวน์โหลดใน Colab) ดู [`docs/COLAB_RUN_TH.md`](docs/COLAB_RUN_TH.md)

## Protocol โดยย่อ
| หัวข้อ | ค่า |
|---|---|
| สินทรัพย์ | ACWI, SHY, XLK, XLV, GLD |
| น้ำหนักอ้างอิง / Alert | 40/20/20/15/5 และ 20/60/10/7.5/2.5 |
| ข้อมูล | เริ่มจากวันแรกที่มีราคาจริงครบ 5 ETF ในปี 2008, 60 ผลตอบแทนแรกเป็น lookback |
| Split (วัน origin) | train ถึง 2017, validation 2018–2019, test 2020–2025, purge target ที่คร่อมขอบ |
| Target | sample SD (ddof=1) ของ r_ref วัน t+1..t+5 × √252 |
| Input | R: 60×5 ETF returns; RV: 60×6 (+ VIX level lag 1 วันทำการ) |
| โมเดล | LSTM, Transformer encoder × {R, RV}; baselines HV20, EWMA(0.94), GARCH(1,1) |
| การฝึก | log-vol MSE, Adam, batch 32, ≤100 epochs, patience 10, grid lr×dropout, seeds 11/22/33 = 48 runs, ensemble 3 seeds |
| Alerts | τ = P90 ของ training targets; cutoff = max F1 บน validation |
| Backtest | signal t → trade t+1, rebalance รายวัน, ต้นทุน 10 bps (5/20 sensitivity), self-financing |

รายละเอียดและตำแหน่งในโค้ด: [`docs/PROTOCOL_TH.md`](docs/PROTOCOL_TH.md) · การตรวจงานวิจัย/repository ที่อ้างอิง: [`docs/SOURCES_AUDIT_TH.md`](docs/SOURCES_AUDIT_TH.md)

## โครงสร้าง
```
configs/protocol.yaml       protocol ที่ล็อกไว้ (smoke.yaml, synthetic_e2e.yaml = overrides สำหรับตรวจโค้ด)
src/volvix/data/            download (raw snapshot + hash), clean (ปฏิทิน, exclusion log), dataset (target, split, scaler)
src/volvix/baselines.py     HV20, EWMA, GARCH
src/volvix/models.py        LSTM, Transformer
src/volvix/training.py      grid 48 runs, checkpoint/resume, selection, ensemble
src/volvix/evaluation.py    metrics, cutoffs, bootstrap, (เสริม) DM/VaR
src/volvix/backtest.py      การบัญชีพอร์ตและต้นทุน
src/volvix/plotting.py      PNG/SVG + CSV ต้นทาง + figure manifest
src/volvix/reporting.py     รายงานคุณภาพข้อมูลและ handover ภาษาไทย
src/volvix/pipeline.py      stages + freeze/test guards
scripts/run_pipeline.py     CLI
scripts/mirror_to_sheets.py mirror ตารางสรุปไป Google Sheets
scripts/make_bundle.py      รวมผลขนาดเล็กเป็น zip
notebooks/colab_run.ipynb   Colab runner (ตรึง commit)
tests/                      correctness tests
```

## รันในเครื่อง (ตรวจโค้ดด้วยข้อมูลสังเคราะห์)
```bash
pip install -r requirements.txt
python -m pytest -q
python scripts/run_pipeline.py --root /tmp/vx --run-name synth --override configs/synthetic_e2e.yaml --snapshot synthetic --stages all
```
ผลจาก `synthetic` ไม่ใช่ผลวิจัย (กราฟมีลายน้ำ SYNTHETIC และ manifest ระบุ `snapshot_synthetic: true`)

## โฟลเดอร์ผลลัพธ์ (บน Google Drive)
```
vix_term_project/
  data/raw/<snapshot_id>/          ไฟล์ดิบ + snapshot_metadata.json (sha256, วันดาวน์โหลด, เวอร์ชันแพ็กเกจ)
  runs/<run_name>/
    config_frozen.yaml  run_manifest.json  environment/
    processed/  dataset/  scalers/  baselines/  training/<run_id>/
    frozen/freeze.json  validation/  test/  backtest/  figures/  reports/
```

## การใช้ Generative AI
โค้ดและเอกสารใน repository นี้จัดทำโดยใช้ Claude (Anthropic) ช่วยเขียนตาม protocol ในร่างรายงาน ผู้จัดทำต้องตรวจโค้ด รันการทดลอง และรับผิดชอบความถูกต้องของผลและข้อสรุปเอง
