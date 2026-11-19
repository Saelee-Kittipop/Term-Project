# คู่มือรันบน Google Colab และรายการผลลัพธ์ที่ต้องส่งกลับ

## ก่อนเริ่ม
1. เปิด notebook จากลิงก์ **Open in Colab** ใน README (ลิงก์ผูกกับ commit ที่ตรึงไว้)
2. Runtime → Change runtime type → **T4 GPU**
3. ผลทั้งหมดจะอยู่ที่ `MyDrive/vix_term_project/` (ข้อมูลดิบ, checkpoints, ผลลัพธ์) ไม่มีอะไรเก็บใน Git

## ขั้นตอน (รันทีละ cell ตามลำดับ)
| ข้อ | ทำอะไร | เวลาโดยประมาณ | ต้องตรวจอะไร |
|---|---|---|---|
| 0 | mount Drive, clone repo ที่ commit ที่ตรึงไว้, ติดตั้ง arch/yfinance/pandas_market_calendars | 1–2 นาที | ข้อความ `repository pinned at <commit>` |
| 1 | pytest (ข้อมูลสังเคราะห์) | < 1 นาที | ต้องผ่านทั้งหมด ถ้าไม่ผ่านให้หยุดและส่ง log |
| 2 | ตรวจ SHA-256 ของไฟล์ 8 ไฟล์ใน `MyDrive/vix_term_project/input_files/` (Yahoo chart JSON 6 ไฟล์, Cboe `VIX_History.csv`, French RF) เทียบกับไฟล์ของ formal_v1 → สร้าง raw snapshot `FILES_*` แล้วรัน smoke (train/validation เท่านั้น) | 1–3 นาที | ทุกไฟล์ขึ้น `OK`, `sample counts match formal_v1: True`, alignment checks ต้อง `True` ทุกแถว |
| 3 | formal: data + baselines (ใช้ snapshot เดียวกับ smoke) | < 1 นาที | GARCH `converged: true` |
| 4 | train 48 runs | ประมาณ 30–90 นาที (ขึ้นกับ early stopping) | `completed 48` |
| 5 | freeze | < 1 นาที | τ, config ที่เลือก, cutoffs, validation metrics |
| 6 | เปิด test ครั้งเดียว, backtest, กราฟ, handover | 2–5 นาที | ตาราง test และ handover ภาษาไทย |
| 7 | mirror ไป Google Sheets | < 1 นาที | URL ของ spreadsheet |
| 8 | สร้าง `bundle_formal_colab_v1.zip` และดาวน์โหลด | < 1 นาที | ไฟล์ zip |

เวลาข้อ 4 เป็นการประมาณจากการรันบน CPU ในการทดสอบของผู้พัฒนาด้วยข้อมูลสังเคราะห์ ยังไม่ได้วัดบน T4 จริง

## ถ้า Colab หลุดกลางทาง
รันข้อ 0 ใหม่ แล้วรัน cell ที่ค้างซ้ำ: training ข้าม run ที่เสร็จแล้วและทำต่อจาก `last.pt` ของ run ที่ค้าง; stage อื่นอ่านไฟล์จาก Drive เดิม

## ถ้าไม่มีไฟล์ใน input_files หรือ hash ไม่ตรง
อัปโหลดไฟล์จาก `Desktop/Term Project/data` ไปที่ `MyDrive/vix_term_project/input_files/` โดยใช้ชื่อเดิมทุกตัวอักษร และอย่าเปิดหรือแปลงเป็น Google Sheets cell ข้อ 2 จะหยุดถ้า hash ไม่ตรง ส่วนการดาวน์โหลดตรงจาก Yahoo ยังทำได้ด้วย `scripts/download_data.py` (ด้านล่าง) แต่จะเป็นข้อมูลชุดใหม่ที่ห้ามปนกับ formal_v1

## ถ้าดาวน์โหลดข้อมูลไม่สำเร็จ (กรณีใช้ scripts/download_data.py)
Yahoo อาจตอบ 429 (จำกัดอัตรา) โค้ดจะลองใหม่ 5 ครั้งสลับ query1/query2 แล้วหยุดพร้อม error (ไม่สร้างข้อมูลแทน) ให้รอสักครู่แล้วรันข้อ 2 ใหม่ หรือใช้ yfinance แทนด้วย `--override configs/source_yfinance.yaml` หากแหล่งใดเข้าไม่ได้เลย สามารถวางไฟล์ชื่อเดียวกันใน `MyDrive/vix_term_project/manual/` แล้วเพิ่ม `--manual-dir` ในคำสั่ง ไฟล์นั้นจะถูกบันทึกเป็น `source: manual` ใน metadata

## สิ่งที่ต้องส่งกลับ (output bundle)
ส่ง **`bundle_formal_colab_v1.zip`** ไฟล์เดียว (สร้างในข้อ 8; อยู่ที่ `MyDrive/vix_term_project/`) โดยแนบในแชตหรือวางในโฟลเดอร์ Term Project ภายในประกอบด้วย:

- `runs/formal_v1/run_manifest.json` — commit, config hash, data hashes, seeds, เวลารัน, สถานะทุก stage
- `runs/formal_v1/config_frozen.yaml`, `environment/pip_freeze.txt`, `environment/environment.json`
- `runs/formal_v1/processed/*.csv|json` — step log (จำนวนแถวก่อน/หลัง), exclusion log, corporate actions audit, data quality
- `runs/formal_v1/dataset/*.csv|json` — split manifest, alignment checks, ตัวอย่างวันที่
- `runs/formal_v1/scalers/*.json`, `baselines/*.csv|json`
- `runs/formal_v1/training/grid_summary.csv` และ `history.csv`, `status.json` ของทุก run
- `runs/formal_v1/frozen/freeze.json`, `validation/*.csv`
- `runs/formal_v1/test/*.csv` — forecasts, metrics, รายปี, subgroup, per-seed, paired bootstrap, ผลเสริม DM/VaR
- `runs/formal_v1/backtest/*.csv` — NAV รายวัน, น้ำหนัก, ต้นทุน, สรุปผลทุกต้นทุน
- `runs/formal_v1/figures/*.png|svg`, `figures/data/*.csv`, `figure_manifest.csv`
- `runs/formal_v1/reports/data_quality_report_th.md`, `handover_th.md`
- `runs/smoke_v1/...` (ผล smoke), `data/raw/<snapshot>/snapshot_metadata.json`, `pytest_*.log`, `smoke.log`, `formal.log`

Checkpoints (`best.pt`) และไฟล์ parquet อยู่ใน Drive และไม่อยู่ใน bundle เพราะมีขนาดใหญ่ hash ของ checkpoints ที่ใช้จริงอยู่ใน `freeze.json`
