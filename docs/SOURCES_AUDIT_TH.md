# การตรวจงานวิจัยและ repository ก่อนนำโค้ดมาใช้

ตรวจเมื่อ 19 พฤศจิกายน 2569 (commit และ license อ่านจาก GitHub API ณ วันตรวจ) หลักการ: นำโค้ดมาใช้ได้เฉพาะเมื่อ license อนุญาต **และ** นิยาม target กับการป้องกัน leakage ตรงกับ protocol ของเรา

## สรุปผล

**ไม่มีการคัดลอกโค้ดจาก repository วิจัยใดเข้ามาในโครงการนี้** โค้ดทั้งหมดเขียนใหม่ตาม protocol ส่วนที่ใช้จากภายนอกมีเพียงไลบรารีที่ติดตั้งผ่าน pip ตามเงื่อนไข license ของแต่ละตัว

## รายละเอียดราย repository

| Repository / แหล่ง | Commit ที่ตรวจ | License | สิ่งที่ศึกษา | นำมาใช้หรือไม่ | เหตุผลและการตรวจ target/leakage |
|---|---|---|---|---|---|
| [EduardoRamosP/MultiTransformer](https://github.com/EduardoRamosP/MultiTransformer) (Ramos-Pérez et al., 2021) | `a94784e6862615c5031de073fc40796159a435f4` | ไฟล์ LICENSE.txt ระบุทั้ง "Creative Commons Attribution 4.0" และ "CC-BY-NC-ND 4.0" (GitHub: NOASSERTION) | การคำนวณ forward SD (`TrueSDCalculation`), โครง Transformer+GARCH (Keras) | ไม่ใช้ | license กำกวมและมีเงื่อนไข NoDerivatives จึงไม่ดัดแปลงโค้ด; `TrueSDCalculation` คำนวณ `std(returns[i:i+5])` ซึ่งรวมผลตอบแทนวัน i เอง ต่างจาก target t+1..t+5 ของเรา จึงต้องระวังหากเทียบผล; ใช้ log returns ขณะที่เราใช้ simple returns |
| [hugogobato/Forecasting-Realized-Volatility-through-Financial-Turbulence-and-Neural-Networks](https://github.com/hugogobato/Forecasting-Realized-Volatility-through-Financial-Turbulence-and-Neural-Networks) (Souto & Moradi, 2023) | `6f620efd622af72def9fdd6fe3e6fedd7cd634fc` | ไม่มีไฟล์ license (สงวนสิทธิ์ทั้งหมดโดยปริยาย) | แนวคิด ablation ข้อมูลพื้นฐาน vs ข้อมูลความเสี่ยงเพิ่ม (Financial Turbulence) ภายใต้ LSTM เดียวกัน | ไม่ใช้ (ใช้แนวคิดการออกแบบเท่านั้น) | target เป็น realized volatility จาก OHLC (Yang–Zhang) ไม่ใช่ forward 5-day sample SD; ไม่มี license จึงไม่คัดลอกโค้ด |
| [zcakhaa/Deep-Learning-in-Quantitative-Trading](https://github.com/zcakhaa/Deep-Learning-in-Quantitative-Trading) (companion ของหนังสือ Zhang & Zohren, 2025) | `7c4ace779818b433097d14d490f8e1149bd3ec1c` | MIT | `Utilis/early_stopper.py`, Chapter 6 portfolio optimization | ไม่ใช้ (เขียน early stopping เอง) | `EarlyStopping` ใน repo ไม่บันทึก checkpoint เมื่อ epoch แรกเป็นค่าที่ดีที่สุด (ครั้งแรกตั้ง best_loss โดยไม่ save) และไม่คืนน้ำหนัก best ให้อัตโนมัติ; protocol ต้องคืนน้ำหนัก epoch ที่ val ต่ำสุด จึงเขียนใหม่ใน `training.train_one`; Chapter 6 เรียนรู้น้ำหนักพอร์ตโดยตรง ต่างจากการแยก forecast กับกฎลงทุนของเรา |
| [stevearchuleta/riskml-capstone](https://github.com/stevearchuleta/riskml-capstone) (แหล่ง GitHub ที่ sync ใน Project) | `fc1bf2a5fadc5f67e30cfe74a2286c33ab911dab` | Apache-2.0 | `riskml/features/transforms.py` (realized vol), `etl/market_data.py` (yfinance `auto_adjust=True`) | ไม่ใช้ | ใช้ daily **log** returns และ rolling vol ย้อนหลังเป็น feature; ไม่มี forward 5-day target แบบ purge ตามของเรา จึงใช้เป็นตัวอย่างเปรียบเทียบเท่านั้น |

## ไลบรารีที่ใช้เป็น dependency (ไม่ได้คัดลอกโค้ด)

| ไลบรารี | Commit/เวอร์ชันที่ตรวจ | License | ใช้ทำอะไร | การตรวจกับ protocol |
|---|---|---|---|---|
| [arch](https://github.com/bashtage/arch) | `04b6e283fd2e4f77adf02de8343481dbcb8290fc` (main), ทดสอบกับ arch 9.0.0 | NCSA/University of Illinois style (permissive) | ประมาณ GARCH(1,1) และ forecast variance หลาย horizon | ทดสอบใน `tests/test_baselines_and_metrics.py` ว่าค่าที่ได้เท่ากับ recursion มือ √(252/5·Σv)/100 และไม่ใช่ vol ของวันที่ 5 เพียงค่าเดียว |
| [yfinance](https://github.com/ranaroussi/yfinance) | `2492a853aee975b4f39f230ad763298f039e37ae` | Apache-2.0 | ดาวน์โหลดราคา `auto_adjust=True` และ `auto_adjust=False` (เพื่อตรวจ corporate actions) | ไม่ใช้ `repair=True` เพื่อไม่ให้ข้อมูลถูกแก้ไขโดยไม่บันทึก; ตรวจเงินปันผลซ้ำใน `corporate_actions_audit.csv` |
| PyTorch | ตามที่ติดตั้งใน Colab (บันทึกใน `pip_freeze.txt`) | BSD-3-Clause | LSTM, TransformerEncoder | positional encoding เขียนเองตามสูตร Vaswani et al. (2017) |
| scikit-learn | ตามที่ติดตั้ง | BSD-3-Clause | `average_precision_score`, `precision_recall_curve` | AP ใช้ forecast volatility เป็น score ไม่เรียกว่า probability |
| pandas_market_calendars | ตามที่ติดตั้ง | MIT | ปฏิทิน NYSE สำหรับตรวจ missing sessions | — |

## แหล่งข้อมูล

| ข้อมูล | URL | หมายเหตุ |
|---|---|---|
| ราคา ETF | Yahoo Finance chart API v8 ดึงตรง (`query1/query2.finance.yahoo.com/v8/finance/chart/<T>`) ไฟล์ JSON ต้นฉบับเก็บใน snapshot; yfinance เป็นทางเลือก | ไม่ใช่ point-in-time archive; adjusted prices อาจถูกผู้ให้บริการปรับย้อนหลัง จึงเก็บ raw snapshot + hash |
| VIX | https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv | ใช้ CLOSE, lag 1 วันทำการ; เทียบกับ Yahoo `^VIX` ใน `snapshot_metadata.json` |
| RF รายวัน | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip | หน่วยเปอร์เซ็นต์ → หาร 100; ใช้เฉพาะคำนวณ Sharpe |
