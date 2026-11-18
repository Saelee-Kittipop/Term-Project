# คู่มือส่งต่อผลเพื่อเขียนรายงานและทำสไลด์

หลังรันข้อ 6 ใน Colab ระบบสร้าง `runs/formal_v1/reports/handover_th.md` จากไฟล์ผลจริงโดยอัตโนมัติ เอกสารนี้บอกว่าแต่ละตาราง/ภาพในบทที่ 4 ของร่างรายงานมาจากไฟล์ใด

| ส่วนของรายงาน | ไฟล์ต้นทาง (ใน run directory) | หมายเหตุ |
|---|---|---|
| ตารางสถานะการทดลอง (ตัวอย่างสุทธิ, สัดส่วน high risk) | `dataset/split_manifest.csv`, `reports/high_risk_share_by_split.csv` | ตัวอย่างสุทธิหลัง purge |
| สถิติเชิงพรรณนา | `reports/descriptive_stats_by_split.csv`, `reports/data_quality_report_th.md` | returns, target, VIX แยก split |
| ภาพข้อมูล (returns, target, VIX, เส้นแบ่ง, τ) | `figures/F01_data_overview.*`, `figures/data/F01_data_overview.csv` | |
| สหสัมพันธ์ training | `figures/F02_train_return_correlation.*` | |
| ตารางผลการพยากรณ์ (MAE, RMSE, log-RMSE, QLIKE) | `test/test_metrics.csv` | MAE/RMSE หน่วย percentage points |
| ผลต่างจากการเพิ่ม VIX + CI | `test/test_paired_bootstrap.csv` (แถว `primary=True`), `figures/F05_*` | ค่าลบ = model_a ดีกว่า |
| ผลราย seed | `test/test_metrics_per_seed.csv`, `validation/validation_metrics_per_seed.csv` | ห้ามเลือกเฉพาะ seed ที่ดีที่สุด |
| ผลรายปี 2020–2025 | `test/test_metrics_by_year.csv` | |
| ภาพ actual vs forecast | `figures/F04_test_forecast_vs_actual.*` | แกนเดียวกันทุกโมเดล |
| Learning curves, best epoch, params, เวลา, memory | `figures/F03_learning_curves.*`, `training/grid_summary.csv` | |
| การเลือก hyperparameters | `frozen/selection.json`, `figures/F10_validation_grid.*` | |
| ตารางผลการแจ้งเตือน (Precision, Recall, F1, AP, false alerts/100) | `test/test_metrics.csv` คอลัมน์ `alert_*` | τ และ cutoffs ใน `frozen/freeze.json` |
| Confusion matrices, PR curves | `figures/F06_*`, `figures/F07_*` | เส้นอ้างอิง = prevalence |
| Subgroup trailing vol ≤ τ | `test/test_alert_subgroup_low_trailing_vol.csv` | ระบุ n |
| ตารางผลทางเศรษฐศาสตร์ | `backtest/backtest_summary.csv` (`is_base_cost=True`) | รวม static comparator และ sensitivity 5/20 bps |
| NAV/drawdown, น้ำหนัก SHY/หุ้น | `figures/F08_*`, `figures/F09_*`, `backtest/backtest_daily.csv` | เริ่ม NAV 100 เท่ากันทุกกลยุทธ์ |
| ผลเสริม (นอก protocol หลัก) | `test/supplementary_diebold_mariano.csv`, `test/supplementary_var_backtest.csv` | ระบุว่าเป็นผลเสริม |

## โครงสไลด์ที่แนะนำ
1. คำถามวิจัยและพอร์ต (ภาพที่ 1–3 ในร่างรายงาน)
2. ข้อมูลและการป้องกัน leakage (F01, split manifest, alignment checks)
3. โมเดลและการเลือก hyperparameter (F10, F03)
4. RQ1: forecast accuracy (test_metrics, F04)
5. RQ2: ประโยชน์ของ VIX (paired bootstrap, F05)
6. Alerts (F06, F07)
7. RQ3: ผลการลงทุนหลังต้นทุน (backtest_summary, F08, F09)
8. ข้อจำกัดและงานต่อยอด
