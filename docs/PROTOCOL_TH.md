# Protocol ที่ล็อกไว้และการแปลงเป็นโค้ด

เอกสารนี้แสดงว่าข้อกำหนดแต่ละข้อในร่างรายงาน `portfolio_volatility_research_th_2008_core60.docx` (ฉบับปรับปรุง 2) ถูกนำไปใช้ที่ไฟล์ใด ค่าตัวเลขทั้งหมดอยู่ใน `configs/protocol.yaml` และถูกคัดลอกเป็น `config_frozen.yaml` ในทุก run โดยระบบจะไม่ยอมรันต่อถ้า config เปลี่ยนภายใต้ run name เดิม

## 1. ข้อกำหนดหลักและตำแหน่งในโค้ด

| ข้อกำหนด | ค่า | ไฟล์ / ฟังก์ชัน |
|---|---|---|
| สินทรัพย์และลำดับ | ACWI, SHY, XLK, XLV, GLD | `protocol.yaml: data.tickers` |
| แหล่งราคา | Yahoo chart API v8 ดึงตรงผ่าน HTTPS (endpoint ที่ yfinance ใช้); adjusted Close = adjclose ตาม `auto_adjust=True` | `data/yahoo_direct.py`, `data/from_files.parse_yahoo_chart`; ทางเลือก `configs/source_yfinance.yaml` |
| แหล่ง VIX | Cboe CLOSE; Yahoo `^VIX` ใช้ตรวจเทียบ และใช้แทนเฉพาะเมื่อ Cboe ดาวน์โหลดไม่ได้ (บันทึก `vix_source_used`) | `data/from_files.py` |
| น้ำหนักอ้างอิง (Normal) | 40/20/20/15/5 | `portfolio.reference_weights`, ตรวจผลรวม=1 ใน `config.validate_config` |
| น้ำหนัก Alert | 20/60/10/7.5/2.5 (= 50% อ้างอิง + 50% SHY) | `portfolio.alert_weights`, ตรวจสูตรใน `validate_config` |
| พอร์ตเปรียบเทียบไม่เพิ่ม sector | ACWI 75 / SHY 20 / GLD 5 | `portfolio.no_tilt_weights` |
| จุดเริ่มข้อมูล | วันแรกที่มีราคาจริงครบ 5 ETF (ตรวจจาก raw ไม่กำหนดเอง) | `data/clean.py: clean_snapshot → common_start` |
| Lookback เริ่มต้น | 60 ผลตอบแทนแรก → origin แรกคือ position 60 | `data/dataset.py: build_samples`, test `test_first_origin_uses_first_60_returns` |
| Target | sd(r_ref t+1..t+5, ddof=1)×√252 | `dataset.forward_target`, ตรวจเทียบ `rolling(5).std().shift(-5)` |
| Input | R = 60×5 returns, RV = 60×6 (+ VIX ของวัน s−1) | `dataset.feature_matrix`, `clean.py: vix_lag1` |
| Split | train ≤ 2017, validation 2018–2019, test 2020–2025 (ตามวัน origin) | `protocol.yaml: splits` |
| Purge | เก็บ origin เฉพาะเมื่อ target_end อยู่ในช่วงเดียวกัน | `build_samples`, `split_manifest` |
| Scaler | StandardScaler fit บนแถวรายวันที่ไม่ซ้ำในหน้าต่างของ training samples | `dataset.TrainOnlyScaler`, `training.fit_scalers` |
| Baselines | HV20, EWMA λ=0.94 (init = mean r² ของ 60 วันแรก), GARCH(1,1) constant mean Gaussian fit บน training returns | `baselines.py` |
| GARCH 5 วัน | √(252/5 · Σ v̂(t+h)) หน่วย % → /10000 | `baselines.garch_forecast`, test เทียบ recursion มือ |
| Models | LSTM(32)→Dropout→Dense16 ReLU→Dense1 ; Transformer proj32 + PE + 1 encoder block (2 heads, FF64) → GAP → Dense16 → Dense1 | `models.py` |
| Loss | MSE บน z = log(max(y, 1e−6)) | `training.train_one` |
| การฝึก | Adam, batch 32, ≤100 epochs, patience 10, คืนน้ำหนัก epoch ที่ val log-MSE ต่ำสุด | `training.train_one` |
| Grid | lr {1e−3, 3e−4} × dropout {0.1, 0.2} × seeds {11,22,33} × 2 feature sets × 2 architectures = 48 runs | `training.grid_specs` |
| เลือก config | 1 ชุดต่อ architecture จาก mean val log-MSE ข้าม 2 feature sets × 3 seeds; เสมอ → dropout สูงกว่า → lr ต่ำกว่า | `training.select_configs` |
| Ensemble | เฉลี่ย predicted log-vol 3 seeds แล้วจึง exp | `training.ensemble_predict_log` |
| High risk | τ = P90 ของ training targets ที่ label ครบ | `evaluation.tau_from_train`, `pipeline.stage_freeze` |
| Alert cutoff | max F1 บน validation, alert เมื่อ forecast > c, รวมค่าต่ำกว่าค่าต่ำสุด, เสมอ → cutoff สูงสุด, ไม่มี positive → c = τ | `evaluation.select_cutoff` |
| Static comparator | (1−q)·Normal + q·Alert, q = ความถี่ alert บน validation | `stage_freeze` |
| ตรึงก่อน test | `frozen/freeze.json` เก็บ τ, cutoffs, q, selection และ SHA-256 ของ config, data, scalers, baselines, checkpoints 12 ไฟล์ | `stage_freeze`, `verify_freeze` |
| ใช้ test ครั้งเดียว | สร้าง `test/TEST_OPENED.json`; รันซ้ำต้องระบุเหตุผล (แก้ implementation เท่านั้น) และถูกบันทึก | `stage_test` |
| Backtest | signal หลัง close t → ซื้อขาย close t+1 → รับผล t+2; rebalance ทุกวัน; ต้นทุน 10 bps (sensitivity 5/20); หักจาก NAV ด้วย bisection | `backtest.py` |
| ตัวชี้วัดลงทุน | net CAGR, vol, Sharpe (excess เหนือ RF รายวัน), MDD, ระยะฟื้นตัว, turnover/ปี, ต้นทุนสะสม | `backtest.summarize` |

## 2. รายละเอียด implementation ที่ร่างรายงานไม่ได้ระบุ (บันทึกไว้เพื่อความโปร่งใส ไม่ได้เลือกจากผล test)

1. **StandardScaler** ใช้ส่วนเบี่ยงเบนมาตรฐานแบบ population (ddof=0) เหมือน scikit-learn
2. **τ** ใช้ `numpy.quantile` แบบ linear interpolation (ค่าเริ่มต้น)
3. **Output bias** ของชั้นสุดท้ายถูกตั้งเป็นค่าเฉลี่ย z ของ training ก่อนฝึก (`training.init_output_bias_to_train_mean`) เพื่อให้เริ่มที่ระดับ log-vol ที่สมเหตุสมผล ใช้เฉพาะข้อมูล training
4. **Mini-batch shuffle** สุ่มลำดับตัวอย่าง training ทุก epoch ด้วย generator ที่ผูกกับ seed (ไม่ใช่ random split ของเวลา ตัวอย่างยังเป็นหน้าต่างตามเวลาเดิม)
5. **Early stopping** นับว่าดีขึ้นเมื่อ val log-MSE ต่ำกว่าค่าที่ดีที่สุดเดิมอย่างเคร่งครัด
6. **Transformer** ใช้ `nn.TransformerEncoderLayer` (post-LayerNorm, ReLU, dropout ภายใน block) และ dropout หลัง global average pooling, positional encoding แบบ sinusoidal ไม่มี causal mask เพราะทุกแถวในหน้าต่าง ≤ t
7. **ตัวอย่างร่วม (common evaluation set)** ทุกโมเดลและทั้ง R/RV ใช้ origin ชุดเดียวกัน คือ origin ที่หน้าต่าง 60 วันมี returns ครบและ VIX lag ครบ เพื่อให้ ablation ควบคุมตัวอย่างเท่ากัน
8. **EWMA** หากมีผลตอบแทนใช้ไม่ได้ จะไม่อัปเดต variance วันนั้นและบันทึกตำแหน่งไว้ (origin ที่ได้รับผลกระทบถูกตัดออกจากการประเมินอยู่แล้ว)
9. **GARCH** ใช้ `arch` (`rescale=False`, ผลตอบแทน×100) ประมาณบนผลตอบแทนถึงวันทำการสุดท้ายของ 2017 แล้ว `fix(params)` เพื่อกรอง variance ไปข้างหน้า หากไม่ลู่เข้าจะคืนค่า NaN และบันทึก `converged=false` ไม่แทนด้วยโมเดลอื่น
10. **Backtest**: CAGR นับต้นทุนซื้อครั้งแรกใน NAV; daily net returns เริ่มจากวันถัดจากวันซื้อครั้งแรก (N = จำนวนวันผลตอบแทน); turnover รวมการซื้อครั้งแรก; วันสุดท้ายไม่มีการซื้อขาย; ถ้าขาด signal ในวันใด (ไม่ควรเกิดกับข้อมูลที่สะอาด) จะคงสถานะเดิมและบันทึกใน `signal_log.csv`; ถ้าขาดผลตอบแทนในช่วง backtest จะหยุดพร้อม error (ไม่ข้ามวันเงียบ ๆ)
11. **Sharpe** ใช้เฉพาะวันที่มี RF จาก French; จำนวนวันที่ขาดถูกรายงานใน `sharpe_days_missing_rf`
12. **แหล่งราคา**: ร่างรายงานระบุ yfinance แต่โค้ดดึง Yahoo chart API โดยตรงตามที่ผู้จัดทำขอ ข้อมูลมาจาก endpoint เดียวกันและแปลงด้วยกติกาเดียวกับ `auto_adjust=True` ควรเขียนในรายงานว่า "ข้อมูลราคาจาก Yahoo Finance (chart API v8)"
13. **VIX** ใช้คอลัมน์ CLOSE ของไฟล์ Cboe `VIX_History.csv`; RF ใช้คอลัมน์ RF ของ `F-F_Research_Data_Factors_daily` หาร 100
14. **ปฏิทิน** ใช้ NYSE จาก `pandas_market_calendars`; หากไม่มีแพ็กเกจจะใช้ union ของวันที่ ETF และบันทึกแหล่งปฏิทินในรายงานคุณภาพข้อมูล

## 3. ความต่างระหว่าง Project instructions กับร่างรายงาน และการตัดสินใจ

ผู้ใช้กำหนดให้ร่างรายงานเป็น specification หลัก จึงยึดร่างรายงานในจุดที่ขัดกัน

| ประเด็น | Project instructions | ร่างรายงาน (ที่ใช้) | การจัดการ |
|---|---|---|---|
| การแบ่งข้อมูล | Walk-forward | Chronological split เดียว ไม่ refit | ใช้ split เดียวตามรายงาน และ purge 5 วันที่ขอบ |
| Features ของ Model B | VIX level, VIX returns, term structure | VIX level lag 1 วันเท่านั้น | ใช้ VIX level lag 1 ตามรายงาน |
| Features ของ Model A | returns, realized vol, technical stats | ETF returns 60×5 | ใช้ returns ตามรายงาน |
| DM test, VaR, Kupiec, Christoffersen | ต้องรายงาน | ไม่อยู่ใน protocol | คำนวณเป็น **ผลเสริม** ในไฟล์ `supplementary_*` ไม่ใช้เลือกโมเดล |

## 4. ประเภทของการรัน

| run_kind | ข้อมูล | ขั้นตอนที่อนุญาต | ความหมาย |
|---|---|---|---|
| `smoke` | จริงหรือสังเคราะห์ | data, baselines, train (1 config, 1 seed, 2 epochs), freeze | ตรวจว่าโค้ดรันได้บน train/validation เท่านั้น ไม่ใช่ผล |
| `synthetic_e2e` | สังเคราะห์เท่านั้น | ทุกขั้นตอน | ตรวจ code path ครบวงจร ตัวเลขไม่ใช่ผลวิจัย ภาพมีลายน้ำ SYNTHETIC |
| `formal` | จริงเท่านั้น | ทุกขั้นตอน | การทดลองจริงตาม protocol |
