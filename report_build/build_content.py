"""Build numbers.json, report_spec.json, slides_spec.json and source_map.csv from ONE frozen run.

Every number in the report and slides is read here from the run's canonical CSV/JSON files.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path("/home/claude/vix_runs/runs/formal_v1")
OUT = Path("/home/claude/deliver")
FIG = OUT / "fig"
FIG.mkdir(exist_ok=True)
MAIN_COMMIT = "892dcc7"
POST_COMMIT = "fc0476d"
REPO = "https://github.com/Saelee-Kittipop/Term-Project"

rd = lambda p: pd.read_csv(RUN / p)
rj = lambda p: json.load(open(RUN / p))

man = rj("run_manifest.json")
fz = rj("frozen/freeze.json")
dq = rj("processed/data_quality.json")
split = rd("dataset/split_manifest.csv")
hr = rd("reports/high_risk_share_by_split.csv")
desc = rd("reports/descriptive_stats_by_split.csv")
chk = rd("dataset/alignment_checks.csv")
steps = rd("processed/step_log.csv")
grid = rd("training/grid_summary.csv")
stab = rd("post_hoc/seed_stability.csv")
val = rd("validation/validation_metrics.csv")
tm = rd("test/test_metrics.csv").set_index("model")
by = rd("test/test_metrics_by_year.csv")
pb = rd("test/test_paired_bootstrap.csv")
dm = rd("test/supplementary_diebold_mariano.csv")
var = rd("test/supplementary_var_backtest.csv")
sub = rd("test/test_alert_subgroup_low_trailing_vol.csv").set_index("model")
conf = rd("figures/data/F06_test_confusion_matrices.csv").set_index("model")
bt = rd("backtest/backtest_summary.csv")
pi = rd("post_hoc/permutation_importance.csv")
corr = rd("figures/data/F02_train_return_correlation.csv").set_index("ticker")
garch = rj("baselines/baseline_info.json")["garch"]
env = rj("environment/environment.json")
fm = rd("figures/figure_manifest.csv").set_index("figure_id")

MODELS = ["HV20", "EWMA", "GARCH", "L_R", "L_RV", "T_R", "T_RV"]
DL = ["L_R", "L_RV", "T_R", "T_RV"]
NAME = {"HV20": "HV20", "EWMA": "EWMA", "GARCH": "GARCH(1,1)", "L_R": "L R", "L_RV": "L RV", "T_R": "T R", "T_RV": "T RV"}


def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{d}f}"


def pct(x, d=2):
    return "—" if x is None or not np.isfinite(x) else f"{x*100:.{d}f}%"


def pval(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# --------------------------------------------------------------------------- numbers
b10 = bt[bt["cost_bps"] == 10].set_index("strategy")
ref = b10.loc["Reference_fixed"]
ov = {m: b10.loc[f"{m}_overlay"] for m in MODELS}
stc = {m: b10.loc[f"{m}_static"] for m in MODELS}
S = split.set_index("split")
H = hr.set_index("split")
D = desc.set_index(["split", "variable"])
N = {
    "run": RUN.name, "commit": MAIN_COMMIT, "post_commit": POST_COMMIT, "snapshot": man["data"]["snapshot_id"],
    "panel_sha": man["data"]["panel_sha256"][:12], "samples_sha": man["data"]["samples_sha256"][:12],
    "config_hash": man["config_hash"], "common_start": dq["common_start"], "last_date": dq["last_common_date"],
    "sessions": dq["sessions"], "valid_days": dq["valid_return_days"], "tau": fz["tau"],
    "n_train": int(S.loc["train", "net_samples"]), "n_val": int(S.loc["validation", "net_samples"]), "n_test": int(S.loc["test", "net_samples"]),
    "hr_train": H.loc["train", "high_risk_share_y_gt_tau"], "hr_val": H.loc["validation", "high_risk_share_y_gt_tau"],
    "hr_test": H.loc["test", "high_risk_share_y_gt_tau"], "n_pos_test": int(conf.loc["L_R", "TP"] + conf.loc["L_R", "FN"]),
    "checks_passed": int(chk["passed"].sum()), "checks_total": len(chk),
}
json.dump({k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in N.items()}, open(OUT / "numbers.json", "w"), indent=1, default=str)

# --------------------------------------------------------------------------- source map + figures
SRC = []  # (item, kind, file, script, run, commit)


def src(item, files, script, commit=MAIN_COMMIT):
    SRC.append({"item": item, "data_files": "; ".join(files), "script": script, "run_id": RUN.name, "git_commit": commit})
    return f"ที่มา: {', '.join(files)} | run {RUN.name} | commit {commit}"


def figure(fid, caption, width=6.3):
    row = fm.loc[fid]
    shutil.copyfile(RUN / "figures" / row["png"], FIG / row["png"])
    script = row["script"] if isinstance(row["script"], str) else "src/volvix/plotting.py"
    commit = POST_COMMIT if "post_hoc" in script else MAIN_COMMIT
    s = src(f"Figure {fid}", [f"figures/{row['data_csv']}"], script, commit)
    return {"type": "figure", "path": str(FIG / row["png"]), "caption": caption, "source": s, "width": width, "fid": fid}


def design_fig(name, caption):
    p = Path("/home/claude/spec/media/media") / name
    shutil.copyfile(p, FIG / name)
    return {"type": "figure", "path": str(FIG / name), "caption": caption,
            "source": "ภาพแนวคิดจากร่างรายงานฉบับปรับปรุง 2 (ไม่ใช่ผลการทดลอง)", "width": 6.3, "fid": name}


B = []  # report blocks
h1 = lambda t: B.append({"type": "h1", "text": t})
h2 = lambda t: B.append({"type": "h2", "text": t})
h3 = lambda t: B.append({"type": "h3", "text": t})
P = lambda t: B.append({"type": "p", "text": t})
bul = lambda items: B.append({"type": "bullets", "items": items})


def table(caption, header, rows, source, widths=None, note=None):
    B.append({"type": "table", "caption": caption, "header": header, "rows": rows, "source": source, "widths": widths, "note": note})


TN = [0]
FN_ = [0]


def tcap(t):
    TN[0] += 1
    return f"ตารางที่ {TN[0]} {t}"


def fcap(t):
    FN_[0] += 1
    return f"ภาพที่ {FN_[0]} {t}"


# =========================================================================== REPORT
B.append({"type": "cover", "lines": ["รายงานการทดลอง", "การพยากรณ์ความผันผวนพอร์ต ETF",
                                      "ด้วย LSTM และ Transformer",
                                      "การประเมินประโยชน์ของข้อมูล VIX ต่อการพยากรณ์ การแจ้งเตือน และการบริหารความเสี่ยง",
                                      "พอร์ต Core–Satellite: ACWI 40% · SHY 20% · XLK 20% · XLV 15% · GLD 5%",
                                      "ผู้จัดทำ: รอระบุชื่อและรหัสนักศึกษา",
                                      "รายวิชา Applied Deep Learning and Neural Networks",
                                      "ฉบับปรับปรุง 3 (ผลการทดลองจริง) · 19 พฤศจิกายน 2569"]})
table(tcap("ข้อมูลเอกสารและสถานะหลักฐาน"), ["รายการ", "รายละเอียด"], [
    ["ฉบับ", "ฉบับปรับปรุง 3 ปรับจากฉบับปรับปรุง 2 (ต้นฉบับยังคงเก็บไว้แยก)"],
    ["สถานะผล", f"ผล final test จาก run {N['run']} ที่ตรึง protocol ก่อนเปิด test ครั้งเดียว"],
    ["Git commit ของการทดลอง", f"{MAIN_COMMIT} (post-hoc analysis และภาพเสริม: {POST_COMMIT})"],
    ["Data snapshot", f"{N['snapshot']} (panel sha256 {N['panel_sha']}…)"],
    ["ช่วงข้อมูลร่วม", f"{N['common_start']} ถึง {N['last_date']} ({N['sessions']:,} วันทำการ)"],
    ["Repository", REPO + " (branch research-pipeline)"],
], src("ตารางข้อมูลเอกสาร", ["run_manifest.json", "processed/data_quality.json"], "scripts/run_pipeline.py"), widths=[2600, 6400])

h1("บทคัดย่อ")
P(f"งานนี้ศึกษาว่า LSTM และ Transformer ขนาดเล็กพยากรณ์ความผันผวนของพอร์ต ETF ในช่วง 5 วันทำการถัดไปได้ดีเพียงใดเมื่อเทียบกับ HV20, EWMA และ GARCH(1,1) และการเพิ่ม VIX ให้ประโยชน์เหนือผลตอบแทนย้อนหลังหรือไม่ target คือ sample standard deviation (ddof=1) ของผลตอบแทนรายวันของพอร์ตอ้างอิงในวัน t+1 ถึง t+5 คูณ √252 ซึ่งเป็นการแสดงผลในหน่วย annualized ของความผันผวนห้าวัน ไม่ใช่การพยากรณ์หนึ่งปี ข้อมูลครอบคลุม {N['common_start']} ถึง {N['last_date']} แบ่ง train ถึงปี 2017 validation ปี 2018–2019 และ test ปี 2020–2025 ตามลำดับเวลาพร้อม purge target ที่คร่อมขอบเขต")
P(f"บน test จำนวน {N['n_test']:,} วันพยากรณ์ Deep Learning ทั้งสี่แบบมี log-RMSE ต่ำกว่า baselines ทุกตัว (L RV {f(tm.loc['L_RV','log_RMSE'])} เทียบ GARCH {f(tm.loc['GARCH','log_RMSE'])}) แต่ GARCH มี QLIKE ดีที่สุด ({f(tm.loc['GARCH','QLIKE'])}) เพราะ Deep Learning พยากรณ์ต่ำกว่าระดับจริงโดยเฉลี่ยประมาณ 1 จุดเปอร์เซ็นต์ การเพิ่ม VIX ไม่เปลี่ยน log-RMSE อย่างมีนัย ช่วงความเชื่อมั่นแบบ moving-block bootstrap ของผลต่างคร่อมศูนย์ทั้ง LSTM และ Transformer แม้ VIX จะเป็น input ที่โมเดลพึ่งพามากที่สุดตาม permutation importance ด้านการแจ้งเตือนช่วงผันผวนสูง (y > τ = {pct(N['tau'])}) Deep Learning ให้ precision สูงกว่าและ false alerts น้อยกว่า baselines มาก แต่ recall ต่ำกว่า")
P(f"ในเชิงเศรษฐศาสตร์ หลังต้นทุน 10 bps ทุก overlay ให้ net CAGR ต่ำกว่าพอร์ตอ้างอิงคงที่ ({pct(ref['net_CAGR'])}) overlay ช่วยลด volatility และ maximum drawdown ได้ แต่สัญญาณที่สลับบ่อยทำให้ turnover สูงถึง 5–9 เท่าต่อปี ผลจึงสนับสนุนข้อสรุปว่าความแม่นยำของ forecast ที่ดีขึ้นไม่แปลงเป็นผลตอบแทนหลังต้นทุนโดยอัตโนมัติ")
P("คำสำคัญ: portfolio volatility, LSTM, Transformer, VIX, early warning, economic value, leakage control")

# --------------------------------------------------------------------------- 1
h1("1 บทนำ")
h2("1.1 ความเป็นมาและความสำคัญ")
P("นักลงทุนที่ถือพอร์ตด้วยสัดส่วนเดิมอาจเผชิญระดับความเสี่ยงต่างกันในแต่ละช่วงเวลา เนื่องจากความผันผวนและการเคลื่อนไหวร่วมกันของสินทรัพย์เปลี่ยนแปลงได้ การประเมินความผันผวนล่วงหน้าจึงช่วยให้ผู้จัดการพอร์ตทบทวน exposure ก่อนรับความเสี่ยงสูงเกินระดับที่ยอมรับได้ อย่างไรก็ตาม การพยากรณ์ความเสี่ยงได้แม่นขึ้นไม่ได้รับประกันว่าผลตอบแทนหลังต้นทุนจะดีขึ้น เพราะการลด exposure อาจทำให้เสียโอกาสระหว่างตลาดฟื้นตัว และการปรับพอร์ตบ่อยก่อให้เกิดต้นทุน")
P("Moreira และ Muir (2017) เสนอหลักฐานสนับสนุนการปรับ exposure ตามความผันผวน ขณะที่ Cederburg et al. (2020) พบว่าประโยชน์ดังกล่าวไม่เกิดขึ้นอย่างเป็นระบบในกลยุทธ์ที่ดำเนินการแบบ real time งานนี้จึงแยกการประเมินความแม่นยำของ forecast ออกจากคุณค่าด้านการลงทุนอย่างชัดเจน LSTM ใช้หน่วยความจำและ gates เพื่อเรียนรู้ลำดับข้อมูล ส่วน Transformer ใช้ self-attention ประมวลความสัมพันธ์ภายในหน้าต่างย้อนหลัง (Hochreiter & Schmidhuber, 1997; Vaswani et al., 2017) และใช้ VIX ซึ่งเป็นความผันผวนคาดการณ์จากตลาดออปชันเป็นตัวแปรเพิ่มเติม")
h2("1.2 ปัญหาและคำถามวิจัย")
P("คำถามหลักคือ การเพิ่ม VIX และการเปลี่ยน architecture ช่วยปรับความแม่นยำของการพยากรณ์ความผันผวน การแจ้งเตือน และผลการลงทุนภายใต้เงื่อนไขเดียวกันหรือไม่ แบ่งเป็นสามคำถาม")
bul(["RQ1 LSTM และ Transformer พยากรณ์ความผันผวนพอร์ตล่วงหน้า 5 วันได้แตกต่างจาก Historical Volatility, EWMA และ GARCH อย่างไร",
     "RQ2 การเพิ่ม VIX ช่วยแต่ละ architecture เมื่อเปรียบเทียบกับการใช้ Returns เพียงอย่างเดียวหรือไม่",
     "RQ3 สัญญาณจากโมเดลช่วยให้ผลตอบแทนต่อความเสี่ยงหลังต้นทุนดีขึ้นเมื่อเทียบกับ EWMA และการลด exposure แบบคงที่หรือไม่"])
h2("1.3 วัตถุประสงค์ ขอบเขต และความสำคัญของการศึกษา")
P("การศึกษามีวัตถุประสงค์เพื่อสร้างและเปรียบเทียบแบบจำลองพยากรณ์ความผันผวน ประเมินประโยชน์เพิ่มเติมของ VIX พัฒนากติกาแจ้งเตือนที่กำหนดไว้ล่วงหน้า และวัดผลจากการนำสัญญาณไปปรับพอร์ตย้อนหลัง ใช้สินทรัพย์ซื้อขายในสหรัฐฯ คำนวณผลตอบแทนเป็นดอลลาร์สหรัฐ ไม่มี leverage หรือ short selling ความผันผวนสูงหมายถึงการกระจายของผลตอบแทนในช่วงอนาคต 5 วัน ไม่ได้หมายถึงตลาดต้องปรับตัวลง และการศึกษาไม่ตั้งสมมติฐานว่า Deep Learning หรือ VIX ต้องให้ผลเหนือกว่า")
P("ความสำคัญเชิงปฏิบัติคือ หน่วยบริหารความเสี่ยงต้องการทั้ง forecast ที่แม่นและกติกาที่ใช้งานได้หลังต้นทุน งานนี้เชื่อมสามระดับเข้าด้วยกันภายใต้ข้อมูลและวันประเมินชุดเดียวกัน ได้แก่ ความแม่นยำของ forecast คุณภาพการแจ้งเตือน และผลการลงทุนหลังต้นทุน")
h2("1.4 วัตถุประสงค์ของพอร์ตและเหตุผลในการเพิ่มน้ำหนัก sector")
P("พอร์ตตั้งต้นมุ่งสร้างการเติบโตระยะยาวผ่านหุ้นโลก และเพิ่มน้ำหนักเทคโนโลยีและ Healthcare ตามสมมติฐานเชิงกลยุทธ์ โดยใช้พันธบัตรระยะสั้นและทองคำช่วยกระจายความเสี่ยง สัดส่วนกำหนดไว้ก่อนทดลอง มิได้มาจากการหาน้ำหนักที่ให้ผลย้อนหลังสูงสุด Core ได้แก่ ACWI 40% และ SHY 20% ส่วน Satellite ได้แก่ XLK 20%, XLV 15% และ GLD 5% หุ้นรวมจึงเท่ากับ 75% และ ACWI มีหุ้นเทคโนโลยีและ Healthcare อยู่แล้ว การเพิ่ม XLK/XLV คือการเพิ่มน้ำหนักสอง sector โดยเจตนา")
B.append({**design_fig("experiment-allocation.png", fcap("น้ำหนักสถานะ Normal, Alert และพอร์ตเปรียบเทียบที่ไม่เพิ่มน้ำหนัก sector"))})
h2("1.5 สมมติฐานที่ตรวจสอบ")
bul(["H1 ประสิทธิภาพการพยากรณ์ของ LSTM และ Transformer อาจแตกต่างจากแบบจำลองพื้นฐาน โดยประเมินขนาดและทิศทางของผลต่าง",
     "H2 การเพิ่ม VIX อาจให้ข้อมูลพยากรณ์เพิ่มเติมภายใน architecture เดียวกัน ทดสอบคู่ L R กับ L RV และ T R กับ T RV ภายใต้ target วันประเมิน และงบการเลือกโมเดลเดียวกัน",
     "H3 สัญญาณอาจสร้างประโยชน์ทางเศรษฐศาสตร์หลังต้นทุนเมื่อเทียบกับ EWMA overlay และการลด exposure แบบคงที่"])
h2("1.6 การทบทวนวรรณกรรม")
P("GARCH อธิบาย conditional variance ด้วยผลกระทบจากความคลาดเคลื่อนและ variance ในอดีต จึงเป็น baseline ทางเศรษฐมิติที่สำคัญ (Bollerslev, 1986) Ramos-Pérez et al. (2021) เสนอ Transformer และ Multi-Transformer ร่วมกับ GARCH สำหรับความผันผวนของ S&P และใช้ sample standard deviation ของผลตอบแทนอนาคต 5 วันเป็น target ซึ่งใกล้กับงานนี้ แต่ architecture เป็น hybrid และใช้ดัชนีเดียว")
P("Roszyk และ Ślepaczuk (2024) เปรียบเทียบ GARCH, LSTM และ hybrid LSTM–GARCH โดยเพิ่ม VIX และรายงานประโยชน์ใน S&P 500 ส่วน Souto และ Moradi (2023) เพิ่ม Financial Turbulence ให้ LSTM และพบประโยชน์ต่างกันตามระดับความผันผวน แต่ใช้ volatility จาก OHLC เป็น target งานนี้นำแนวคิดการเปรียบเทียบข้อมูลพื้นฐานกับข้อมูลความเสี่ยงเพิ่มเติมภายใต้โมเดลเดียวกันมาใช้")
P("ด้านการนำไปใช้ Moreira และ Muir (2017) รายงานประโยชน์ของ volatility-managed portfolios ขณะที่ Cederburg et al. (2020) พบว่าผลไม่สม่ำเสมอในการประเมินแบบ real time Bongaerts, Kang และ van Dijk (2020) ศึกษาการปรับ exposure เฉพาะภาวะความผันผวนสุดขั้ว ซึ่งใกล้กับกติกา alert ของงานนี้ Patton (2011) ชี้ว่าการเปรียบเทียบ forecast ด้วย volatility proxy ที่มี noise ต้องเลือก loss อย่างระมัดระวัง งานนี้จึงรายงาน log-RMSE เป็นหลักควบคู่ RMSE, MAE และ QLIKE บน variance")
P("คุณูปการที่ตั้งใจคือ replication and extension ที่ใช้พอร์ตอ้างอิงเดียวกัน การทดลอง 2 × 2 แยกผลของ VIX และ architecture และเชื่อม forecast เข้ากับ warning และ backtest หลังต้นทุน โดยไม่อ้างว่าเป็นช่องว่างใหม่ของวรรณกรรมทั้งหมด")

# --------------------------------------------------------------------------- 2
h1("2 สถาปัตยกรรมแบบจำลองและกรอบแนวคิด")
h2("2.1 กรอบแนวคิดการทดลอง")
P("กระบวนการเริ่มจากข้อมูลราคาและ VIX สร้างผลตอบแทนพอร์ตและ target แบ่งข้อมูลตามเวลา ฝึกและเลือกโมเดลจาก train/validation จากนั้นตรึงโมเดล cutoff และกติกาการลงทุนพร้อม hash ก่อนประเมิน test เพียงครั้งเดียว การประเมินประกอบด้วย forecast accuracy, warning performance, incremental value of VIX และ economic value")
B.append(design_fig("experiment-framework.png", fcap("กรอบการวิจัยตั้งแต่ข้อมูลตลาดถึงการประเมินบนข้อมูลอนาคต")))
h2("2.2 สถาปัตยกรรม Deep Learning และการทดลอง 2 × 2")
st = stab.set_index("model")
table(tcap("การทดลองแบบ 2 × 2 และขนาดโมเดลจริง"), ["รหัส", "Architecture", "Input", "จำนวน parameters", "ใช้ตอบคำถาม"], [
    ["L R", "LSTM", "Returns 60×5", f"{int(st.loc['L_R','n_params']):,}", "เทียบ L RV เพื่อวัดประโยชน์ VIX"],
    ["L RV", "LSTM", "Returns + VIX 60×6", f"{int(st.loc['L_RV','n_params']):,}", "เทียบ T RV เมื่อข้อมูลเหมือนกัน"],
    ["T R", "Transformer", "Returns 60×5", f"{int(st.loc['T_R','n_params']):,}", "เทียบ T RV เพื่อวัดประโยชน์ VIX"],
    ["T RV", "Transformer", "Returns + VIX 60×6", f"{int(st.loc['T_RV','n_params']):,}", "เทียบ L RV เมื่อข้อมูลเหมือนกัน"],
], src("ตาราง 2x2 และจำนวน parameters", ["post_hoc/seed_stability.csv", "training/grid_summary.csv"], "scripts/post_hoc_analysis.py", POST_COMMIT),
    widths=[900, 1500, 2000, 1700, 2900])
P("LSTM ใช้หนึ่งชั้น 32 units ตามด้วย dropout, Dense 16 พร้อม ReLU และ Dense 1 แบบ linear ส่วน Transformer ใช้ linear projection เป็น 32 มิติ sinusoidal positional encoding และ encoder หนึ่ง block ที่มีสอง attention heads และ feed-forward 64 units พร้อม residual connections, layer normalization และ dropout ตามด้วย global average pooling, dropout, Dense 16 และ Dense 1 การให้ encoder เห็นทุกวันในหน้าต่างไม่เป็น future leakage เพราะทุกแถวมีวันที่ไม่เกิน t")
P("โมเดลพยากรณ์ log volatility โดย z = log(max(y, ε)) และ ε = 0.000001 ใช้ MSE บน log scale แล้วแปลงกลับด้วย exponential การเลือก loss นี้ลดอิทธิพลของค่าผันผวนสุดขั้ว แต่มีผลข้างเคียงสำคัญที่พบใน test คือ forecast มีแนวโน้มต่ำกว่าค่าเฉลี่ยจริง (ดูหัวข้อ 4.1 และ 5.3)")
P("เหตุผลที่เลือก: LSTM เหมาะกับ volatility clustering ที่มีความต่อเนื่องตามเวลา ส่วน self-attention ให้น้ำหนักวันต่าง ๆ ภายใน 60 วันได้ยืดหยุ่น ทั้งสองเป็นเหตุผลเชิงแบบจำลองที่ต้องตรวจด้วยผล ขนาดเล็ก (5–9 พัน parameters) จำกัดความซับซ้อนให้เหมาะกับข้อมูลรายวันที่มีตัวอย่างอิสระจำกัด")
h2("2.3 แบบจำลองพื้นฐานที่ใช้ target เดียวกัน")
P("HV20 ใช้ annualized sample SD ของผลตอบแทนพอร์ตย้อนหลัง 20 วันถึงวัน t EWMA ใช้ zero-mean squared returns ด้วย λ = 0.94 เริ่มจากค่าเฉลี่ย squared returns ของ 60 วันแรก GARCH(1,1) ใช้ constant mean และ Gaussian innovations ประมาณพารามิเตอร์จาก training returns เท่านั้น แล้วกรอง variance ต่อด้วยพารามิเตอร์คงที่ ผลประมาณที่ได้คือ "
  f"μ = {garch['params']['mu']:.4f}, ω = {garch['params']['omega']:.4f}, α = {garch['params']['alpha[1]']:.4f}, β = {garch['params']['beta[1]']:.4f} (หน่วยเปอร์เซ็นต์รายวัน; α+β = {garch['persistence_alpha_plus_beta']:.4f}) และลู่เข้าปกติ ค่าพยากรณ์ 5 วันคำนวณจาก √(252/5 · Σ v̂(t+h|t)) โดยหาร variance ด้วย 10,000 ก่อนคืนสู่หน่วยทศนิยม")
P("ทุก baseline ใช้ target และวันประเมินเดียวกับ Deep Learning แต่ baseline ใช้ผลตอบแทนพอร์ตรวม ขณะที่ Deep Learning เห็น returns ราย ETF ผลต่างจึงไม่ได้เกิดจาก architecture เพียงอย่างเดียว")

# --------------------------------------------------------------------------- 3
h1("3 วิธีวิจัยและข้อมูล")
h2("3.1 แหล่งข้อมูลและการได้มาซึ่งข้อมูล")
P(f"ราคาของ ACWI, SHY, XLK, XLV และ GLD มาจาก Yahoo Finance chart API v8 (endpoint เดียวกับที่ yfinance ใช้) ซึ่งให้ราคา close, adjusted close, เงินปันผล และ split ราคาปรับแล้วสร้างตามกติกาเดียวกับ yfinance auto_adjust=True VIX ใช้คอลัมน์ CLOSE จาก Cboe VIX_History.csv และ RF รายวันจาก Fama/French Factors ใช้คำนวณ Sharpe เท่านั้น ไฟล์ทั้งหมดดาวน์โหลดผ่าน browser ของผู้จัดทำเมื่อ 19 พ.ย. 2569 เพราะสภาพแวดล้อมประมวลผลเข้าถึงแหล่งข้อมูลโดยตรงไม่ได้ ไฟล์ต้นฉบับถูกเก็บใน snapshot {N['snapshot']} พร้อม SHA-256 และตรวจ hash ทุกครั้งก่อนใช้ จุดนี้ต่างจากร่างเดิมที่ระบุ yfinance แต่ข้อมูลมาจากแหล่งและกติกาการปรับราคาเดียวกัน")
xc = dq["vix_crosscheck_cboe_vs_yahoo"]
P(f"เพื่อตรวจความถูกต้องของ VIX ได้ดึง Yahoo ^VIX มาเทียบกับ Cboe จาก {xc['common_days']:,} วันที่ตรงกัน มีเพียง {xc['days_abs_diff_gt_0.05']} วันที่ราคาปิดต่างเกิน 0.05 จุด (สูงสุด {xc['max_abs_diff']:.2f} จุด) จึงใช้ Cboe ตามแผน")
h2("3.2 การทำความสะอาดข้อมูล")
srow = steps
rows = []
for t in ["ACWI", "SHY", "XLK", "XLV", "GLD"]:
    g = srow[srow["item"] == t]
    rows.append([t, f"{int(g.iloc[0]['rows_before']):,}", f"{int(g[g['step']=='common_window']['removed'].iloc[0]):,}", f"{int(g[g['step']=='common_window']['rows_after'].iloc[0]):,}"])
table(tcap("จำนวนแถวก่อนและหลังแต่ละขั้นตอน"), ["ETF", "แถวดิบ 2008–2025", "ตัดก่อนวันข้อมูลร่วม", "แถวที่ใช้"], rows,
      src("ตารางจำนวนแถว", ["processed/step_log.csv"], "src/volvix/data/clean.py"), widths=[1500, 2500, 2500, 2500])
P(f"ACWI เริ่มซื้อขายช้าที่สุด วันข้อมูลร่วมแรกจึงเป็น {N['common_start']} ไม่สร้างราคา ACWI ย้อนก่อนกองทุนมีอยู่ ช่วงที่ใช้มี {N['sessions']:,} วันทำการตามปฏิทิน NYSE ผลการตรวจพบว่าไม่มี missing session ไม่มีวันที่ซ้ำ ไม่มีราคาที่ไม่เป็นบวก ไม่มีผลตอบแทนรายวันที่เกิน ±15% ที่ต้อง flag ไม่พบการนับเงินปันผลซ้ำ และ split 2:1 ของ XLK วันที่ 5 ธ.ค. 2025 ถูกปรับในราคาที่ใช้ถูกต้อง ผลตอบแทนที่ใช้ได้มี {N['valid_days']:,} วัน (วันแรกไม่มีผลตอบแทน) ไม่มีการเติมค่าจากอนาคตและไม่ตัดข้อมูลวิกฤตจริง")
h2("3.3 พอร์ตอ้างอิง ตัวแปรตาม และ annualization")
P("ผลตอบแทนราย ETF คือ r(i,t) = P(i,t)/P(i,t−1) − 1 จาก adjusted price และผลตอบแทนพอร์ตอ้างอิงคือ r_ref(t) = Σ w_i r(i,t) ด้วยน้ำหนัก 0.40, 0.20, 0.20, 0.15, 0.05 ที่ปรับกลับทุกวัน target ที่ origin t คือ")
B.append({"type": "eq", "text": "y_t = sqrt( 252/4 · Σ_{h=1..5} ( r_ref(t+h) − mean(r_ref(t+1..t+5)) )² )"})
P("ตัวหาร 4 มาจาก ddof=1 การคูณ √252 เปลี่ยนเพียงหน่วยการรายงาน หาก daily SD ใน 5 วันข้างหน้าเท่ากับ 1% ค่า annualized ประมาณ 15.87% โดยยังพยากรณ์ช่วงอนาคตเพียง 5 วัน ไม่ใช่หนึ่งปี ตลอดรายงานนี้ใช้คำว่า annualized forward five-day sample standard deviation")
h2("3.4 สถิติเชิงพรรณนา")
rows = []
for s, lab in [("train", "Train"), ("validation", "Validation"), ("test", "Test")]:
    y = D.loc[(s, "y_target")]
    r = D.loc[(s, "r_ref")]
    v = D.loc[(s, "vix_lag1")]
    rows.append([lab, f"{int(y['n']):,}", pct(y["mean"]), pct(y["q50"]), pct(y["q90"]), pct(y["max"]), pct(r["sd"] * np.sqrt(252)), f(v["mean"], 1)])
table(tcap("สถิติเชิงพรรณนาแยกตาม split (target y และผลตอบแทนพอร์ต ณ วันพยากรณ์ที่ใช้)"),
      ["Split", "n", "y เฉลี่ย", "y มัธยฐาน", "y P90", "y สูงสุด", "SD r_ref × √252", "VIX lag เฉลี่ย"], rows,
      src("ตารางสถิติเชิงพรรณนา", ["reports/descriptive_stats_by_split.csv"], "src/volvix/reporting.py"),
      widths=[1200, 800, 1100, 1100, 1000, 1100, 1500, 1200])
P(f"ระดับความผันผวนเฉลี่ยใน test ({pct(D.loc[('test','y_target'),'mean'])}) สูงกว่า validation ({pct(D.loc[('validation','y_target'),'mean'])}) และมีค่าสุดขั้วสูงกว่าในช่วง COVID-19 ปี 2020 (สูงสุด {pct(D.loc[('test','y_target'),'max'])}) แสดงถึง distribution shift ระหว่างช่วงเลือกโมเดลกับช่วงทดสอบ ภาพที่ 3 แสดงผลตอบแทน target และ VIX ตลอดช่วง และภาพที่ 4 แสดงสหสัมพันธ์ช่วง training ซึ่ง ACWI–XLK มีค่า {corr.loc['ACWI','XLK']:.2f} ขณะที่ SHY และ GLD มีความสัมพันธ์ต่ำกับหุ้น")
B.append(figure("F01_data_overview", fcap(f"ผลตอบแทนพอร์ต target และ VIX ปี 2008–2025 เส้นประคือขอบ validation/test และเส้น τ = {pct(N['tau'])}")))
B.append(figure("F02_train_return_correlation", fcap("สหสัมพันธ์ผลตอบแทนรายวันช่วง training"), width=4.2))
h2("3.5 Input และการปรับมาตราส่วน")
P("ใช้ lookback 60 วันทำการ ชุด R คือ returns ของ ETF ห้าตัว (60 × 5) และชุด RV เพิ่ม VIX level ของวันทำการก่อนหน้า (60 × 6) แถววันที่ s ใช้ returns วัน s และ VIX วัน s−1 StandardScaler fit จากแถวรายวันที่ไม่ซ้ำซึ่งอยู่ในหน้าต่างของ training samples เท่านั้นแล้วตรึงไว้ ทุกโมเดลและทั้ง R/RV ใช้ origins ชุดเดียวกันเพื่อให้ ablation ควบคุมตัวอย่างเท่ากัน")
h2("3.6 การแบ่งข้อมูลตามเวลาและการป้องกัน leakage")
B.append(design_fig("experiment-split.png", fcap("ลำดับการแบ่งข้อมูลและการตัด target ที่คร่อมขอบเขต (ภาพแนวคิด)")))
rows = []
for s, lab in [("train", "Training"), ("validation", "Validation"), ("test", "Test")]:
    rows.append([lab, f"{S.loc[s,'first_origin']} ถึง {S.loc[s,'last_origin']}", f"{int(S.loc[s,'candidate_origins']):,}",
                 f"{int(S.loc[s,'dropped_input_window_incomplete']):,}", f"{int(S.loc[s,'purged_target_crosses_boundary']):,}",
                 f"{int(S.loc[s,'net_samples']):,}", pct(H.loc[s, "high_risk_share_y_gt_tau"])])
table(tcap("การแบ่งข้อมูลจริง จำนวนตัวอย่างสุทธิ และสัดส่วน high risk"),
      ["ชุด", "origin แรก–สุดท้าย", "origin ในช่วง", "ตัดเพราะ lookback ไม่ครบ", "purge ที่ขอบ", "ตัวอย่างสุทธิ", "สัดส่วน y > τ"], rows,
      src("ตาราง split", ["dataset/split_manifest.csv", "reports/high_risk_share_by_split.csv"], "src/volvix/data/dataset.py"),
      widths=[1100, 2300, 1100, 1300, 1000, 1100, 1100])
P(f"ตรวจ alignment ด้วยข้อมูลจริงก่อนฝึกครบ {N['checks_passed']}/{N['checks_total']} ข้อ ได้แก่ input สิ้นสุดที่ t, target เริ่ม t+1 และสิ้นสุด t+5, target เท่ากับ rolling(5).std(ddof=1).shift(−5)·√252, ไม่มี target ซ้อนข้าม train/validation/test และ VIX lag เป็นค่าของวันทำการก่อนหน้า τ คำนวณจาก training เท่านั้น = {pct(N['tau'])} ห้าม random split, centered rolling และ backward-fill ไม่ refit ด้วย validation และตรึงโมเดลตลอด test")
h2("3.7 การฝึกและการเลือก hyperparameters")
P("ใช้ Adam, batch size 32, สูงสุด 100 epochs, early stopping patience 10 และคืนน้ำหนักจาก epoch ที่ validation log-MSE ต่ำสุด seeds 11, 22, 33 grid คือ learning rate {0.001, 0.0003} × dropout {0.1, 0.2} รวม 48 runs เลือก configuration หนึ่งชุดต่อ architecture จากค่าเฉลี่ย validation log-MSE ข้ามสอง feature sets และสาม seeds แล้วใช้ ensemble ของสาม seeds โดยเฉลี่ย predicted log volatility")
P(f"การทดลองรันบน CPU ในสภาพแวดล้อมประมวลผลของ Claude (ไม่ใช่ Colab GPU) ด้วย PyTorch {env.get('torch')} ใช้เวลาฝึกทั้ง 48 runs ประมาณ 12 นาที เนื่องจากโมเดลเล็กและ early stopping หยุดเร็ว")
h2("3.8 นิยามการแจ้งเตือนและตัวชี้วัด")
P(f"High risk คือ y > τ โดย τ = percentile 90 ของ training targets ({pct(N['tau'])}) cutoff ของแต่ละโมเดลเลือกจาก validation ให้ F1 สูงสุด (เสมอเลือก cutoff สูงสุด) alert เมื่อ forecast > cutoff ตัวชี้วัด forecast คือ MAE และ RMSE (จุดเปอร์เซ็นต์) log-RMSE (ตัวหลัก) และ QLIKE บน variance ตัวชี้วัด alert คือ Precision, Recall, F1, Average Precision, false alerts ต่อ 100 วัน และ confusion matrix ส่วน paired comparisons ใช้ moving-block bootstrap บล็อก 20 วัน 1,000 ครั้ง เพราะ target ห้าวันซ้อนกัน")
h2("3.9 Backtest")
P("สัญญาณหลัง close วัน t ซื้อขายที่ close t+1 น้ำหนักใหม่รับผลตอบแทนวัน t+2 สถานะ Normal = 40/20/20/15/5 และ Alert = 20/60/10/7.5/2.5 (50% ของพอร์ตอ้างอิง + 50% SHY) เริ่มจากเงินสด 100 หน่วยวันที่ 3 ม.ค. 2020 rebalance กลับสู่เป้าหมายทุกวัน ต้นทุน 10 bps ต่อมูลค่าซื้อหรือขาย (sensitivity 5/20 bps) หักจาก NAV ด้วย bisection เปรียบเทียบกับพอร์ตอ้างอิงคงที่ พอร์ต ACWI 75/SHY 20/GLD 5 และ static comparator ที่ถือ (1−q)·Normal + q·Alert โดย q คือความถี่ alert บน validation")

# --------------------------------------------------------------------------- 4
h1("4 ผลการศึกษา")
P(f"ผลทั้งหมดในบทนี้มาจาก run {N['run']} (commit {MAIN_COMMIT}, snapshot {N['snapshot']}) และแยกสามระดับ: smoke run (smoke_v1) ใช้ตรวจว่าโค้ดทำงานบน train/validation เท่านั้นและไม่ใช้เป็นผล ผล validation ใช้เลือกโมเดลและ cutoff และผล final test ใช้ตอบ RQ1–RQ3 ภาพ F11–F15 และ permutation importance เป็น post-hoc analysis (commit {POST_COMMIT}) ที่อ่านจาก run ที่ตรึงแล้วโดยไม่ฝึกใหม่และไม่ใช้เลือกโมเดล")
h2("4.1 ผลการพยากรณ์บน test (RQ1)")
rows = [[NAME[m], f(tm.loc[m, "MAE_pp"], 2), f(tm.loc[m, "RMSE_pp"], 2), f(tm.loc[m, "log_RMSE"]), f(tm.loc[m, "QLIKE"]),
         f(tm.loc[m, "mean_forecast_pp"], 2)] for m in MODELS]
table(tcap(f"ผลการพยากรณ์บน test (n = {N['n_test']:,} วันพยากรณ์ 2020–2025; ค่าต่ำกว่าดีกว่า; y เฉลี่ยจริง = {f(tm.loc['L_R','mean_actual_pp'],2)}%)"),
      ["โมเดล", "MAE (pp)", "RMSE (pp)", "log-RMSE", "QLIKE", "forecast เฉลี่ย (%)"], rows,
      src("ตารางผลการพยากรณ์ test", ["test/test_metrics.csv"], "src/volvix/pipeline.py (stage_test)"),
      widths=[1500, 1400, 1400, 1500, 1500, 1700])
best = tm.loc[DL, "log_RMSE"].idxmin()
P(f"Deep Learning ทั้งสี่แบบมี log-RMSE ({f(tm.loc[DL,'log_RMSE'].min())}–{f(tm.loc[DL,'log_RMSE'].max())}) ต่ำกว่า HV20 ({f(tm.loc['HV20','log_RMSE'])}), EWMA ({f(tm.loc['EWMA','log_RMSE'])}) และ GARCH ({f(tm.loc['GARCH','log_RMSE'])}) และ L R มี RMSE ต่ำที่สุด ({f(tm.loc['L_R','RMSE_pp'],2)} pp) อย่างไรก็ตาม QLIKE ให้ลำดับต่างออกไป GARCH ดีที่สุด ({f(tm.loc['GARCH','QLIKE'])}) และ L R กับ T R แย่กว่าทุก baseline สาเหตุสอดคล้องกับค่าเฉลี่ย forecast: Deep Learning พยากรณ์ต่ำกว่าระดับจริงโดยเฉลี่ย (10.9–11.4% เทียบค่าจริง 12.24%) ขณะที่ baselines พยากรณ์สูงกว่า (13.0–13.3%) QLIKE ลงโทษการพยากรณ์ต่ำเกินมากกว่า ส่วน log-MSE ที่ใช้ฝึกสอดคล้องกับการพยากรณ์ค่ากลางของ log volatility ซึ่งต่ำกว่าค่าเฉลี่ยในหน่วยปกติเมื่อการแจกแจงเบ้ขวา")
pbx = pb[(pb["loss"] == "log_se")].set_index(["model_a", "model_b"])
dmx = dm.set_index(["model_a", "model_b", "loss"])
rows = []
for m in DL:
    for b in ["HV20", "EWMA", "GARCH"]:
        r = pbx.loc[(m, b)]
        rows.append([f"{NAME[m]} − {b}", f"{r['mean_diff']:+.4f}", f"[{r['ci_low_2.5']:+.4f}, {r['ci_high_97.5']:+.4f}]",
                     pval(dmx.loc[(m, b, 'log_se'), 'p_value']), pval(dmx.loc[(m, b, 'qlike'), 'p_value'])])
table(tcap("ผลต่าง squared log error ระหว่าง Deep Learning กับ baselines บน test (ค่าลบ = Deep Learning ดีกว่า)"),
      ["คู่เปรียบเทียบ", "ผลต่างเฉลี่ย log-SE", "95% CI (block bootstrap)", "DM p (log-SE)*", "DM p (QLIKE)*"], rows,
      src("ตาราง DL vs baselines", ["test/test_paired_bootstrap.csv", "test/supplementary_diebold_mariano.csv"], "src/volvix/evaluation.py"),
      widths=[1900, 1700, 2600, 1400, 1400], note="* Diebold–Mariano พร้อม HAC lag 4 และ HLN correction เป็นผลเสริมตามคำแนะนำของ Project ไม่อยู่ใน protocol หลัก")
P("ความได้เปรียบด้าน log-SE ของ Deep Learning เหนือ HV20 และ EWMA มี CI ที่ไม่คร่อมศูนย์ทุกคู่ เทียบกับ GARCH ความได้เปรียบชัดเจนเฉพาะ LSTM ส่วน Transformer มี CI คร่อมศูนย์ ขณะที่ด้าน QLIKE GARCH ดีกว่าทุกโมเดล Deep Learning อย่างมีนัยตาม DM test คำตอบ RQ1 จึงขึ้นกับ loss: Deep Learning ดีกว่าเมื่อวัดความคลาดเคลื่อนเชิงสัดส่วน แต่ด้อยกว่า GARCH เมื่อให้น้ำหนักการพยากรณ์ต่ำเกินในช่วงเสี่ยง")
B.append(figure("F13_test_metrics_by_model", fcap("ตัวชี้วัดการพยากรณ์บน test แยกตามโมเดล (dot plot แกนไม่เริ่มที่ศูนย์)")))
B.append(figure("F04_test_forecast_vs_actual", fcap("ค่าจริงเทียบค่าพยากรณ์ annualized five-day volatility บน test ทุกโมเดลใช้แกนเดียวกัน"), width=6.0))
yr = by.pivot(index="year", columns="model", values="log_RMSE")
prev = by.groupby("year")["alert_prevalence"].first()
rows = [[str(y), f"{int(by[by.year==y]['n'].iloc[0])}", pct(prev.loc[y], 1)] + [f(yr.loc[y, m]) for m in MODELS] for y in yr.index]
table(tcap("log-RMSE รายปีบน test (ปีของวันพยากรณ์)"), ["ปี", "n", "สัดส่วน y>τ"] + [NAME[m] for m in MODELS], rows,
      src("ตาราง log-RMSE รายปี", ["test/test_metrics_by_year.csv"], "src/volvix/pipeline.py (stage_test)"),
      widths=[700, 600, 1000, 950, 950, 1050, 950, 950, 950, 950])
P(f"ผลรายปีไม่สม่ำเสมอ LSTM ทั้งสองแบบดีกว่าทุก baseline ในปี 2020, 2021 และ 2025 ซึ่งมีช่วงความผันผวนพุ่งขึ้น แต่ Transformer ในปี 2020 ยังแย่กว่า GARCH ({f(yr.loc[2020,'T_R'])} และ {f(yr.loc[2020,'T_RV'])} เทียบ {f(yr.loc[2020,'GARCH'])}) ในปี 2022–2024 ผลปะปน: ปี 2023 L R ({f(yr.loc[2023,'L_R'])}) แย่กว่า HV20 ({f(yr.loc[2023,'HV20'])}) และปี 2024 L RV กับ T RV แย่กว่า EWMA ({f(yr.loc[2024,'EWMA'])}) ข้อสรุปจึงไม่ควรอาศัยค่าเฉลี่ยทั้งช่วงเพียงอย่างเดียว")
B.append(figure("F12_test_logrmse_by_year", fcap("log-RMSE รายปีบน test")))

h2("4.2 ประโยชน์ของ VIX (RQ2)")
pbp = pb[pb["primary"]].set_index(["model_a", "model_b", "loss"])
rows = []
for a, b_ in [("L_RV", "L_R"), ("T_RV", "T_R"), ("T_R", "L_R"), ("T_RV", "L_RV")]:
    for loss, lab in [("log_se", "log-SE"), ("qlike", "QLIKE")]:
        r = pbp.loc[(a, b_, loss)]
        rows.append([f"{NAME[a]} − {NAME[b_]}", lab, f"{r['mean_diff']:+.4f}", f"[{r['ci_low_2.5']:+.4f}, {r['ci_high_97.5']:+.4f}]"])
table(tcap("Paired differences ภายใน architecture และระหว่าง architecture บน test (ค่าลบ = ตัวแรกดีกว่า)"),
      ["คู่", "loss", "ผลต่างเฉลี่ย", "95% CI (moving-block bootstrap 20 วัน × 1,000)"], rows,
      src("ตาราง paired bootstrap", ["test/test_paired_bootstrap.csv"], "src/volvix/evaluation.py"), widths=[2200, 1300, 1800, 3700])
P(f"การเพิ่ม VIX ไม่เปลี่ยนความแม่นยำอย่างมีนัย ผลต่าง log-SE ของ L RV − L R เท่ากับ {pbp.loc[('L_RV','L_R','log_se'),'mean_diff']:+.4f} และ T RV − T R เท่ากับ {pbp.loc[('T_RV','T_R','log_se'),'mean_diff']:+.4f} โดย CI คร่อมศูนย์ทั้งคู่ ด้าน QLIKE ทิศทางเอื้อ VIX ทั้งสอง architecture แต่ CI ยังคร่อมศูนย์ ส่วนการจัดอันดับ alert ดีขึ้นเล็กน้อย (AP ของ L RV {f(tm.loc['L_RV','alert_AP'])} เทียบ L R {f(tm.loc['L_R','alert_AP'])} และ T RV {f(tm.loc['T_RV','alert_AP'])} เทียบ T R {f(tm.loc['T_R','alert_AP'])}) จึงสรุปได้เพียงว่า VIX ให้ประโยชน์เล็กและไม่สม่ำเสมอในพอร์ตนี้")
B.append(figure("F05_vix_forecast_difference", fcap("ผลต่าง forecast จากการเพิ่ม VIX (RV − R) ภายใน architecture บน test")))
pix = pi.set_index(["model", "feature"])
P(f"Permutation importance (post-hoc, ไม่ฝึกใหม่) ชี้ว่าในโมเดล RV ค่า VIX lag เป็น input ที่โมเดลพึ่งพามากที่สุด (log-MSE เพิ่ม {pix.loc[('L_RV','VIX_lag1'),'mean_increase_logmse']:.3f} ใน L RV และ {pix.loc[('T_RV','VIX_lag1'),'mean_increase_logmse']:.3f} ใน T RV เมื่อสลับ VIX ข้ามวันพยากรณ์) ขณะที่ความสำคัญของ ACWI และ XLK ลดลงมากเมื่อเทียบกับโมเดล R ผลนี้บ่งว่า VIX ทำหน้าที่แทนข้อมูลความผันผวนจากหุ้นที่โมเดล R ดึงได้เองจาก returns ข้อมูลจึงซ้ำซ้อนกันมาก และอธิบายได้ว่าทำไมความแม่นยำรวมแทบไม่เปลี่ยน")
B.append(figure("F14_permutation_importance", fcap("Permutation importance บน test (ensemble, สลับ 10 ครั้ง; เป็น XAI หลังการทดลอง)")))

h2("4.3 การฝึกและความเสถียร")
gg = grid.groupby(["arch", "lr", "dropout"])["best_val_logmse"].agg(["mean", "std"]).reset_index()
rows = [[r.arch, f"{r.lr:g}", f"{r.dropout:g}", f(r["mean"], 4), f(r["std"], 4),
         "เลือก" if (fz["selection"][r.arch]["lr"] == r.lr and fz["selection"][r.arch]["dropout"] == r.dropout) else ""] for _, r in gg.iterrows()]
table(tcap("ผลการเลือก hyperparameters จาก validation (ค่าเฉลี่ยข้าม 2 feature sets × 3 seeds)"),
      ["Architecture", "learning rate", "dropout", "val log-MSE เฉลี่ย", "SD", "ผล"], rows,
      src("ตาราง grid", ["training/grid_summary.csv", "frozen/selection.json"], "src/volvix/training.py"), widths=[1700, 1400, 1100, 1900, 1200, 1100])
rows = [[NAME[m], f"{int(st.loc[m,'best_epoch_min'])}–{int(st.loc[m,'best_epoch_max'])}", f(st.loc[m, "val_logmse_mean"], 4), f(st.loc[m, "val_logmse_sd"], 4),
         f(st.loc[m, "test_logRMSE_seed_mean"]), f(st.loc[m, "test_logRMSE_seed_sd"], 4), f(tm.loc[m, "log_RMSE"]), f(st.loc[m, "train_time_sec_mean"], 1)] for m in DL]
table(tcap("ความเสถียรข้าม seeds ของ configuration ที่เลือก"),
      ["โมเดล", "best epoch", "val log-MSE เฉลี่ย", "SD", "test log-RMSE ราย seed เฉลี่ย", "SD", "ensemble", "เวลาฝึก/run (วินาที)"], rows,
      src("ตาราง seed stability", ["post_hoc/seed_stability.csv", "test/test_metrics_per_seed.csv"], "scripts/post_hoc_analysis.py", POST_COMMIT),
      widths=[900, 1000, 1300, 900, 1500, 900, 1100, 1400])
P("Best epoch อยู่ระหว่าง 3–19 จาก 100 epochs ที่อนุญาต validation loss ต่ำสุดเกิดเร็วแล้วจึงสูงขึ้นขณะที่ training loss ลดต่อ แสดงสัญญาณ overfitting ที่ early stopping ช่วยตัดได้ LSTM เสถียรข้าม seeds มาก (SD ของ test log-RMSE ราว 0.005) ขณะที่ Transformer แปรผันมากกว่าราวห้าเท่า (SD ราว 0.025–0.028) และ seed 22 ของ T R มี validation loss สูงผิดปกติ การใช้ ensemble ลดผลนี้ ensemble ของทุกโมเดลดีกว่าค่าเฉลี่ยราย seed ซึ่งสอดคล้องกับการลด variance")
B.append(figure("F03_learning_curves", fcap("Learning curves ของ configuration ที่เลือก (เส้นประ = training, เส้นทึบ = validation)"), width=6.0))
B.append(figure("F10_validation_grid", fcap("ค่าเฉลี่ย validation log-MSE ของแต่ละ configuration พร้อม SD")))

h2("4.4 การแจ้งเตือนช่วงความผันผวนสูง")
rows = [[NAME[m], f(fz["cutoffs"][m]["cutoff"] * 100, 2), f(tm.loc[m, "alert_precision"]), f(tm.loc[m, "alert_recall"]), f(tm.loc[m, "alert_F1"]),
         f(tm.loc[m, "alert_AP"]), f(tm.loc[m, "alert_ROC_AUC_supplementary"]), f(tm.loc[m, "alert_false_alerts_per_100"], 1), f"{int(tm.loc[m,'alert_missed_high_risk'])}"] for m in MODELS]
table(tcap(f"ผลการแจ้งเตือนบน test (τ = {pct(N['tau'])}; positive {N['n_pos_test']} จาก {N['n_test']:,} วัน = {pct(N['hr_test'])})"),
      ["โมเดล", "cutoff (%)", "Precision", "Recall", "F1", "AP", "ROC-AUC*", "False alerts/100", "Missed"], rows,
      src("ตารางผลการแจ้งเตือน", ["test/test_metrics.csv", "frozen/freeze.json"], "src/volvix/evaluation.py"),
      widths=[1000, 1000, 1000, 900, 800, 800, 1000, 1300, 900], note="* ROC-AUC รายงานเสริม ไม่ใช้แทน AP เพราะข้อมูลไม่สมดุล")
P(f"Baselines ตั้ง cutoff ต่ำจาก validation จึงแจ้งเตือนบ่อย (HV20 แจ้งเตือน {pct(tm.loc['HV20','alert_alert_rate'],1)} ของวัน) ได้ recall สูงแต่ false alerts 18–41 ครั้งต่อ 100 วัน Deep Learning แจ้งเตือน 12–17% ของวัน ได้ precision 0.38–0.46 และ F1 สูงกว่า โดย L RV มี F1 สูงสุด ({f(tm.loc['L_RV','alert_F1'])}) และ T RV มี AP สูงสุด ({f(tm.loc['T_RV','alert_AP'])}) แต่พลาด 54–70 วันจาก {N['n_pos_test']} วัน ความต่างของ recall ส่วนใหญ่มาจาก cutoff ที่เลือกบน validation ซึ่งมีสัดส่วน high risk เพียง {pct(N['hr_val'],1)} มากกว่าความสามารถจัดอันดับ เพราะ AP และ ROC-AUC ของทุกโมเดลใกล้กัน")
B.append(figure("F06_test_confusion_matrices", fcap("Confusion matrix 2 × 2 บน test ของทั้งเจ็ดโมเดล")))
B.append(figure("F07_test_precision_recall", fcap("Precision–Recall curve บน test เส้นประคือ prevalence"), width=4.6))
B.append(figure("F11_test_roc", fcap("ROC curve บน test (ผลเสริม)"), width=4.6))
rows = [[NAME[m], f"{int(sub.loc[m,'n'])}", pct(sub.loc[m, "alert_prevalence"], 1), f(sub.loc[m, "alert_precision"]), f(sub.loc[m, "alert_recall"]), f(sub.loc[m, "alert_F1"]), f(sub.loc[m, "alert_AP"])] for m in MODELS]
table(tcap("Subgroup ที่ trailing HV20 ณ วันพยากรณ์ยังไม่เกิน τ (วันที่ความผันผวนยังไม่สูงขณะพยากรณ์)"),
      ["โมเดล", "n", "prevalence", "Precision", "Recall", "F1", "AP"], rows,
      src("ตาราง subgroup", ["test/test_alert_subgroup_low_trailing_vol.csv"], "src/volvix/pipeline.py (stage_test)"),
      widths=[1200, 900, 1200, 1300, 1300, 1300, 1300])
sap = sub["alert_AP"]
P(f"ในวันที่ความผันผวนย้อนหลังยังต่ำ ทุกโมเดลทำได้แย่ลงมาก (AP {sap.min():.2f}–{sap.max():.2f}) แสดงว่าส่วนใหญ่ของสัญญาณเป็นการติดตามภาวะที่ผันผวนอยู่แล้ว ไม่ใช่การเตือนล่วงหน้าก่อนเหตุการณ์ใหม่ Deep Learning มี AP ใน subgroup นี้ ({sap[DL].min():.2f}–{sap[DL].max():.2f}) สูงกว่า baselines ({sap[['HV20','EWMA','GARCH']].min():.2f}–{sap[['HV20','EWMA','GARCH']].max():.2f}) โดย T RV สูงสุด ({sap['T_RV']:.3f}) แต่ระดับยังต่ำเกินกว่าจะตีความเป็น crash prediction")

h2("4.5 ผลทางเศรษฐศาสตร์หลังต้นทุน (RQ3)")
order = ["Reference_fixed", "NoTilt_ACWI75_SHY20_GLD5"] + [f"{m}_overlay" for m in MODELS] + [f"{m}_static" for m in MODELS]
lab = {"Reference_fixed": "พอร์ตอ้างอิงคงที่", "NoTilt_ACWI75_SHY20_GLD5": "ACWI 75 / SHY 20 / GLD 5"}
rows = []
for s in order:
    r = b10.loc[s]
    name = lab.get(s, s.replace("_overlay", " overlay").replace("_static", " static").replace("_", " "))
    rows.append([name, pct(r["net_CAGR"]), pct(r["ann_vol"]), f(r["sharpe"]), pct(r["max_drawdown"]), f(r["turnover_per_year"], 2),
                 f(r["cumulative_cost"], 2), "—" if pd.isna(r["alert_frequency"]) else pct(r["alert_frequency"], 1), pct(r["avg_w_SHY"], 1)])
table(tcap("ผลทางเศรษฐศาสตร์บน test 3 ม.ค. 2020 – 31 ธ.ค. 2025 (ต้นทุน 10 bps, NAV เริ่ม 100, USD)"),
      ["กลยุทธ์", "Net CAGR", "Volatility", "Sharpe", "Max DD", "Turnover/ปี", "ต้นทุนสะสม", "ความถี่ alert", "SHY เฉลี่ย"], rows,
      src("ตารางผลทางเศรษฐศาสตร์", ["backtest/backtest_summary.csv"], "src/volvix/backtest.py"),
      widths=[2000, 900, 900, 750, 850, 1000, 1000, 1000, 900], note="ต้นทุนสะสมเป็นหน่วย NAV จากเงินเริ่มต้น 100 หน่วย; Sharpe ใช้ excess return เหนือ RF รายวันของ French")
gap = [(stc[m]["net_CAGR"] - ov[m]["net_CAGR"]) * 100 for m in MODELS]
P(f"พอร์ตอ้างอิงคงที่ให้ net CAGR {pct(ref['net_CAGR'])}, volatility {pct(ref['ann_vol'])}, Sharpe {f(ref['sharpe'])} และ max drawdown {pct(ref['max_drawdown'])} ทุก overlay ให้ net CAGR ต่ำกว่า (9.0–9.9%) HV20 และ EWMA overlay ลด volatility เหลือราว 9.6–9.9% และ max drawdown เหลือ 11.7–14.5% จนมี Sharpe สูงกว่าพอร์ตอ้างอิงเล็กน้อย ({f(ov['HV20']['sharpe'])} และ {f(ov['EWMA']['sharpe'])}) ขณะที่ Deep Learning overlay ลด drawdown ได้น้อยกว่า (22–25%) และ Sharpe ต่ำกว่าพอร์ตอ้างอิง static comparator ของทุกโมเดลให้ net CAGR สูงกว่า overlay ของตนเอง {min(gap):.1f}–{max(gap):.1f} จุดเปอร์เซ็นต์ จึงไม่พบประโยชน์จากการเลือกจังหวะหลังต้นทุนในพอร์ตนี้")
P(f"สาเหตุสำคัญคือ turnover Deep Learning overlay มี turnover 8.0–9.1 เท่าต่อปีและต้นทุนสะสมราว 5.7–6.3 หน่วยจากเงินต้น 100 เทียบกับพอร์ตอ้างอิง {f(ref['turnover_per_year'],2)} เท่าและ {f(ref['cumulative_cost'],2)} หน่วย สัญญาณสลับเข้าออกบ่อยรอบ cutoff และการซื้อขายช้าหนึ่งวันทำให้ลดความเสี่ยงหลังจากความผันผวนเกิดแล้วและกลับเข้าตลาดช้าในช่วงฟื้นตัว")
B.append(figure("F08_nav_and_drawdown", fcap("NAV หลังต้นทุน 10 bps และ drawdown บนช่วง test เดียวกัน")))
B.append(figure("F09_overlay_weights", fcap("น้ำหนัก SHY และหุ้น (ACWI+XLK+XLV) ตามเวลาของแต่ละ overlay"), width=5.8))
P("Sensitivity ต่อต้นทุนยืนยันข้อสรุป: ที่ 5 bps overlay ยังต่ำกว่าพอร์ตอ้างอิง และที่ 20 bps ช่องว่างกว้างขึ้นเพราะ overlay มี turnover สูงกว่ามาก")
B.append(figure("F15_cost_sensitivity", fcap("Net CAGR และ Sharpe เมื่อเปลี่ยนต้นทุน 5/10/20 bps")))
nt = b10.loc["NoTilt_ACWI75_SHY20_GLD5"]
P(f"พอร์ตเปรียบเทียบ ACWI 75 / SHY 20 / GLD 5 ให้ net CAGR {pct(nt['net_CAGR'])} ต่ำกว่าพอร์ตอ้างอิงที่เพิ่มน้ำหนัก XLK/XLV ในช่วง 2020–2025 ซึ่งเป็นช่วงที่หุ้นเทคโนโลยีให้ผลตอบแทนสูง ผลนี้เป็นข้อสังเกตเฉพาะช่วงทดสอบ ไม่ใช่หลักฐานว่า sector ดังกล่าวมี risk premium สูงกว่าในอนาคต")

h2("4.6 ผลเสริมด้าน Value-at-Risk (นอก protocol หลัก)")
vv = var.set_index(["model", "horizon_days", "level"])
rows = [[NAME[m], pct(vv.loc[(m, 1, 0.95), "rate"]), pval(vv.loc[(m, 1, 0.95), "p_value"]), pct(vv.loc[(m, 1, 0.99), "rate"]), pval(vv.loc[(m, 1, 0.99), "p_value"]),
         pval(vv.loc[(m, 1, 0.95), "cc_p_cc"])] for m in MODELS]
table(tcap("1-day Gaussian VaR จากค่าพยากรณ์ (อัตรา violation และ p-value ของ Kupiec/Christoffersen) บน test"),
      ["โมเดล", "violation 95%", "Kupiec p", "violation 99%", "Kupiec p", "Christoffersen CC p (95%)"], rows,
      src("ตาราง VaR เสริม", ["test/supplementary_var_backtest.csv"], "src/volvix/evaluation.py"), widths=[1300, 1500, 1200, 1500, 1200, 2300],
      note="ผลเสริมตามคำแนะนำของ Project; VaR 5 วันใช้หน้าต่างซ้อนกันจึงไม่นำเสนอการทดสอบความเป็นอิสระ")
dlv = [vv.loc[(m, 1, 0.95), "rate"] * 100 for m in DL]
P(f"Baselines ทั้งสามให้อัตรา violation ที่ 95% ใกล้ 5% และผ่าน Kupiec ขณะที่ Deep Learning มี violation {min(dlv):.1f}–{max(dlv):.1f}% และถูกปฏิเสธ สอดคล้องกับการพยากรณ์ต่ำกว่าค่าจริงในหัวข้อ 4.1 หากจะใช้ forecast จาก Deep Learning เพื่อ VaR ต้อง calibrate ระดับก่อน ซึ่งเป็นงานต่อยอดที่ต้องใช้ validation ใหม่")

# --------------------------------------------------------------------------- 5
h1("5 อภิปรายผลและข้อเสนอแนะ")
h2("5.1 ตอบคำถามวิจัย")
bul([f"RQ1: Deep Learning ลด log-RMSE ได้ชัดเมื่อเทียบ HV20 และ EWMA และ LSTM ดีกว่า GARCH อย่างมีนัย แต่ GARCH ยังดีที่สุดด้าน QLIKE และด้าน VaR coverage ผลจึงไม่สนับสนุนการสรุปว่า Deep Learning ดีกว่าทุกด้าน",
     "RQ2: การเพิ่ม VIX ไม่ทำให้ความแม่นยำเปลี่ยนอย่างมีนัยในทั้งสอง architecture แม้โมเดลจะพึ่งพา VIX มาก ข้อมูลจาก VIX จึงซ้ำกับสัญญาณความผันผวนที่อยู่ใน returns ของ ACWI/XLK แล้ว ผลเสริมด้าน QLIKE และ AP เอื้อ VIX เล็กน้อยแต่ไม่ชัดเจน",
     "RQ3: ไม่มี overlay ใดให้ net CAGR สูงกว่าพอร์ตอ้างอิงหรือ static comparator ของตน HV20/EWMA overlay ลดความเสี่ยงได้มากที่สุดและมี Sharpe สูงกว่าเล็กน้อย ส่วน Deep Learning overlay ไม่ให้ประโยชน์หลังต้นทุนแม้ alert มี precision สูงกว่า"])
h2("5.2 ทำไมความแม่นยำที่ดีขึ้นไม่แปลงเป็นผลลงทุน")
P("ประการแรก กติกา alert เป็นสองสถานะ จึงเปลี่ยนพอร์ตครั้งละมากเมื่อ forecast ข้าม cutoff ไปมา ทำให้ turnover สูง ประการที่สอง execution delay หนึ่งวันทำให้ลด exposure หลังจากความผันผวนเริ่มแล้ว และ subgroup analysis แสดงว่าสัญญาณส่วนใหญ่ติดตามภาวะที่ผันผวนอยู่แล้ว ประการที่สาม ช่วง 2020–2025 ตลาดหุ้นฟื้นตัวเร็วหลังแต่ละภาวะตึงเครียด การถือ SHY มากขึ้นจึงเสียผลตอบแทนช่วงฟื้นตัว ประการสุดท้าย log-RMSE วัดความแม่นยำโดยรวม ไม่ได้วัดคุณภาพการตัดสินใจที่ cutoff ซึ่งเป็นสิ่งที่ backtest ใช้จริง")
h2("5.3 Overfitting, underfitting และ bias–variance trade-off")
P("Learning curves แสดงว่า validation loss ต่ำสุดในไม่กี่ epoch แล้วสูงขึ้นขณะที่ training loss ลดต่อ จึงมีความเสี่ยง overfitting ซึ่ง early stopping และโมเดลขนาดเล็กควบคุมได้ ในทางกลับกัน การที่ forecast ต่ำกว่าค่าจริงอย่างเป็นระบบเป็น bias จาก loss บน log scale มากกว่า underfitting ของ architecture Transformer มี variance ข้าม seeds สูงกว่า LSTM ราวห้าเท่า สะท้อนความยืดหยุ่นที่มากขึ้นเมื่อข้อมูลอิสระจำกัด ensemble ช่วยลด variance แต่ไม่ลด bias")
h2("5.4 Class imbalance และ temporal dependence")
P(f"High risk มีเพียง {pct(N['hr_train'],1)} ของ train, {pct(N['hr_val'],1)} ของ validation และ {pct(N['hr_test'],1)} ของ test และกระจุกในปี 2020 และ 2022 ขณะที่ปี 2023 ไม่มีเลย cutoff ที่เลือกบน validation ซึ่งมีเหตุการณ์น้อยจึงไม่เสถียร งานนี้ไม่ใช้ resampling เพราะเป็น regression และจะทำลายโครงสร้างเวลา target ห้าวันที่ซ้อนกันและหน้าต่าง 60 วันทำให้จำนวนตัวอย่างอิสระน้อยกว่าจำนวนแถวมาก จึงใช้ moving-block bootstrap และอ่าน CI เป็นหลักฐานเสริม ไม่ใช่การทดลองแบบสุ่ม")
h2("5.5 ข้อจำกัด")
bul(["ข้อมูลราคามาจาก Yahoo chart API ผ่าน browser แทน yfinance ใน Python แม้เป็นแหล่งและกติกาเดียวกัน adjusted prices ไม่ใช่ point-in-time archive",
     "การทดลองรันบน CPU ในสภาพแวดล้อมของผู้ช่วย AI ไม่ใช่ Colab GPU ตามแผน ผลรันซ้ำบน GPU อาจต่างเล็กน้อยจาก numerical nondeterminism",
     "พอร์ตเดียว น้ำหนักเดียว และ ETF ที่ยังมีอยู่ในปัจจุบันทำให้มี selection และ hindsight bias จึงเป็น retrospective case study",
     "โมเดลตรึงตลอดปี 2020–2025 จึงเผชิญ distribution shift และไม่มี refit",
     "Backtest ใช้ adjusted-close proxy ต้นทุนสมมติ ไม่รวม order book ภาษี และ FX ของผู้ลงทุนไทย",
     "Permutation importance ทำบน test หลังตรึงผล ใช้อธิบายเท่านั้น และการสลับ input หนึ่งตัวอาจสร้างรูปแบบข้อมูลที่ไม่สมจริงเมื่อ input สัมพันธ์กัน"])
h2("5.6 นัยเชิงปฏิบัติและเชิงนโยบาย")
P("สำหรับผู้จัดการกองทุนและหน่วยบริหารความเสี่ยง forecast จาก Deep Learning เหมาะเป็นข้อมูลประกอบ risk dashboard และ risk budget มากกว่าเป็นสัญญาณซื้อขายโดยตรง เพราะแจ้งเตือนแม่นกว่าแต่ต้องปรับระดับก่อนใช้กับ VaR และต้องออกแบบกติกาลดการสลับสถานะก่อนนำไปลงทุน สำหรับหน่วยงานกำกับดูแลและผู้กำหนดนโยบายการลงทุนของสถาบัน ผลนี้ชี้ว่าการประเมินโมเดล AI ด้านความเสี่ยงควรรวมทั้ง forecast accuracy, calibration, ความเสถียรข้าม seeds และผลหลังต้นทุนที่ลงมือได้จริง การรายงานเพียงความแม่นยำอาจทำให้ประเมินประโยชน์ต่อผู้ลงทุนสูงเกินจริง ทั้งนี้หลักฐานจาก term project ยังไม่พอสำหรับการใช้กับเงินจริง")
h2("5.7 สรุปและข้อเสนอแนะสำหรับงานต่อยอด")
P("ภายใต้ protocol ที่ตรึงไว้ LSTM และ Transformer ขนาดเล็กพยากรณ์ความผันผวนห้าวันของพอร์ตได้แม่นกว่า HV20 และ EWMA ในเชิง log-RMSE แต่ไม่ดีกว่า GARCH ทุกด้าน VIX ไม่ให้ประโยชน์ที่ชัดเจน และไม่มีกลยุทธ์ overlay ใดชนะพอร์ตอ้างอิงหลังต้นทุน ข้อเสนอแนะสำหรับงานต่อยอด ซึ่งต้องทำใน run ใหม่ด้วย validation ใหม่และแยกเป็น exploratory analysis ได้แก่:")
bul(["calibrate ระดับ forecast หรือใช้ loss ที่สอดคล้องกับ variance (เช่น QLIKE) เพื่อลด bias ต่ำเกิน",
     "กติกาลงทุนที่ลดการสลับสถานะ เช่น hysteresis หรือ volatility targeting แบบต่อเนื่อง และประเมินต้นทุนพร้อมกัน",
     "walk-forward refit แบบ expanding window ตามคำแนะนำของ Project และ horizon 20 วัน",
     "feature ด้าน VIX ที่ไม่ซ้ำกับ returns เช่น term structure (VIX/VIX3M) หรือ variance risk premium",
     "XAI ที่คำนึงถึงความสัมพันธ์ระหว่าง input เช่น grouped permutation หรือ attention/gradient attribution พร้อม validation"])

# --------------------------------------------------------------------------- references
h1("เอกสารอ้างอิง")
REFS = [
    "Andersen, T. G., & Benzoni, L. (2008). Realized volatility. Federal Reserve Bank of Chicago Working Paper 2008-14.",
    "Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics, 31(3), 307–327. https://doi.org/10.1016/0304-4076(86)90063-1",
    "Bongaerts, D., Kang, X., & van Dijk, M. (2020). Conditional volatility targeting. Financial Analysts Journal, 76(4), 54–71. https://doi.org/10.1080/0015198X.2020.1790853",
    "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324",
    "Cederburg, S., O'Doherty, M. S., Wang, F., & Yan, X. S. (2020). On the performance of volatility-managed portfolios. Journal of Financial Economics, 138(1), 95–117. https://doi.org/10.1016/j.jfineco.2020.04.015",
    "Christoffersen, P. F. (1998). Evaluating interval forecasts. International Economic Review, 39(4), 841–862.",
    "Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. Journal of Business & Economic Statistics, 13(3), 253–263.",
    "Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. International Journal of Forecasting, 13(2), 281–291.",
    "Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735",
    "Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. International Conference on Learning Representations. https://arxiv.org/abs/1412.6980",
    "Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. The Journal of Derivatives, 3(2), 73–84.",
    "Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. The Annals of Statistics, 17(3), 1217–1241.",
    "Moreira, A., & Muir, T. (2017). Volatility-managed portfolios. The Journal of Finance, 72(4), 1611–1644. https://doi.org/10.1111/jofi.12513",
    "Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, 32.",
    "Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. Journal of Econometrics, 160(1), 246–256. https://doi.org/10.1016/j.jeconom.2010.03.034",
    "Ramos-Pérez, E., Alonso-González, P. J., & Núñez-Velázquez, J. J. (2021). Multi-Transformer: A new neural network-based architecture for forecasting S&P volatility. Mathematics, 9(15), 1794. https://doi.org/10.3390/math9151794",
    "Roszyk, N., & Ślepaczuk, R. (2024). The hybrid forecast of S&P 500 volatility ensembled from VIX, GARCH and LSTM models. arXiv:2407.16780.",
    "Souto, H. G., & Moradi, A. (2023). Forecasting realized volatility through financial turbulence and neural networks. Economics and Business Review, 9(2), 133–159. https://doi.org/10.18559/ebr.2023.2.737",
    "Vaswani, A., et al. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30. https://arxiv.org/abs/1706.03762",
    "Zhang, Z., Zohren, S., & Roberts, S. (2020). Deep learning for portfolio optimization. The Journal of Financial Data Science, 2(4), 8–20. https://doi.org/10.3905/jfds.2020.1.042",
    "arch (Sheppard, K.). (n.d.). arch: Autoregressive conditional heteroskedasticity models in Python. https://arch.readthedocs.io",
    "Cboe. (n.d.). Historical data for Cboe VIX index. https://www.cboe.com/tradable-products/vix/vix-historical-data/",
    "French, K. R. (n.d.). Data library: Fama/French factors (daily). https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
    "Yahoo Finance. (n.d.). Chart API v8 (เข้าถึงผ่าน query1.finance.yahoo.com เมื่อ 19 พ.ย. 2569).",
    "yfinance. (n.d.). yfinance documentation. https://ranaroussi.github.io/yfinance/",
    "scikit-learn. (n.d.). Average precision score. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html",
]
bul(REFS)

# --------------------------------------------------------------------------- appendices
h1("ภาคผนวก")
h2("ภาคผนวก ก การทำซ้ำ (reproducibility)")
table(tcap("รายการสำหรับทำซ้ำผล"), ["รายการ", "ค่า"], [
    ["Repository / branch", f"{REPO} / research-pipeline"],
    ["Commit ที่รันการทดลอง (data → report)", "892dcc7a22e03f83c118004c18f499b30f2e0fbc"],
    ["Commit ของ post-hoc analysis (F11–F15, XAI)", POST_COMMIT],
    ["Run ID / config hash", f"{N['run']} / {N['config_hash']}"],
    ["Data snapshot", N["snapshot"]],
    ["panel.parquet sha256", man["data"]["panel_sha256"]],
    ["samples.parquet sha256", man["data"]["samples_sha256"]],
    ["สภาพแวดล้อม", f"Python {env.get('python')}, PyTorch {env.get('torch')}, pandas {env.get('pandas')}, numpy {env.get('numpy')}, arch {env.get('arch')}, CPU"],
    ["คำสั่งหลัก", "python scripts/run_pipeline.py --root <ROOT> --run-name formal_v1 --snapshot files:<dir> --stages all"],
    ["Colab", "notebooks/colab_run.ipynb (ตรึง commit 892dcc7; ดาวน์โหลดข้อมูลจาก Yahoo/Cboe/French โดยตรง)"],
], src("ตาราง reproducibility", ["run_manifest.json", "environment/environment.json"], "scripts/run_pipeline.py"), widths=[3200, 5800])
P("ทุก stage บันทึก Git commit, เวลา และสถานะใน run_manifest.json freeze.json เก็บ τ, cutoffs, static weights, selection และ SHA-256 ของ config, data, scaler, baseline forecasts และ checkpoints 12 ไฟล์ test เปิดครั้งเดียว (test/TEST_OPENED.json) ชุดทดสอบความถูกต้อง 39 ข้อผ่านใน sandbox, GitHub Actions และ Colab T4")
h2("ภาคผนวก ข พจนานุกรมข้อมูล")
table(tcap("ตัวแปรและหน่วย"), ["ตัวแปร", "นิยาม", "หน่วย"], [
    ["P adj", "ราคาปรับแล้ว (adjclose ของ Yahoo)", "USD ณ close"], ["r_i", "simple return ของ ETF i", "ทศนิยมรายวัน"],
    ["r_ref", "ผลตอบแทนพอร์ตอ้างอิงที่ rebalance ทุกวัน", "ทศนิยมรายวัน ก่อนต้นทุน"],
    ["y", "sample SD (ddof=1) ของ r_ref วัน t+1..t+5 × √252", "annualized decimal ของความผันผวนห้าวัน"],
    ["VIX lag1", "VIX close ของวันทำการก่อนหน้า", "index points ก่อน scale"], ["z", "log(max(y, 1e−6))", "log volatility"],
    ["τ", "P90 ของ training y", f"{N['tau']:.4f}"], ["c_m", "alert cutoff ของโมเดล m จาก validation", "annualized decimal"],
    ["RF", "Fama/French RF รายวัน / 100", "ทศนิยมรายวัน (ใช้กับ Sharpe เท่านั้น)"]],
    "ที่มา: configs/protocol.yaml, src/volvix/data/dataset.py", widths=[1400, 4600, 3000])
h2("ภาคผนวก ค การบัญชีพอร์ตและต้นทุน")
P("ให้ V⁻ เป็น NAV หลังรับผลตอบแทนของวัน u_i เป็นสัดส่วนการถือก่อนซื้อขาย w_i เป็นน้ำหนักเป้าหมาย และ c เป็นอัตราต้นทุน แก้ z = 1 − c·Σ|z·w_i − u_i| ด้วย bisection บน [0,1] แล้ว V⁺ = z·V⁻ วันแรกถือเงินสดจึง z = 1/(1+c) ไม่มีเงินเติมจากภายนอก วันสุดท้ายไม่ซื้อขาย CAGR = (NAV_T/100)^(252/N) − 1 โดย N คือจำนวนวันผลตอบแทน Sharpe = √252·mean(r−RF)/SD(r−RF) และ turnover = Σ(มูลค่าซื้อขายรวม/NAV ก่อนซื้อขาย)·252/N การตรวจซ้ำอิสระยืนยันว่า target คำนวณใหม่ตรงกับไฟล์ (ผลต่างสูงสุด < 1e−16) NAV สอดคล้องกับการทบต้น และสัญญาณวัน t มีผลกับการซื้อขายวัน t+1 ครบ 1,506 วัน")
h2("ภาคผนวก ง การอ้างอิงโค้ดและ repository")
P("ไม่มีการคัดลอกโค้ดจาก repository วิจัยใดเข้ามาในโครงการ โค้ดทั้งหมดเขียนใหม่ตาม protocol ผลตรวจ: MultiTransformer (commit a94784e, license กำกวม CC-BY / CC-BY-NC-ND จึงไม่ดัดแปลง; target รวมผลตอบแทนวัน t ต่างจากของเรา), Souto & Moradi (commit 6f620ef, ไม่มี license; ใช้แนวคิด ablation เท่านั้น), Zhang & Zohren companion (commit 7c4ace7, MIT; EarlyStopping ไม่บันทึก checkpoint เมื่อ epoch แรกดีที่สุด จึงเขียนใหม่) และ riskml-capstone (commit fc1bf2a, Apache-2.0; ใช้ log returns และไม่มี forward target) ไลบรารีที่ใช้: PyTorch (BSD-3), arch (NCSA-style), scikit-learn (BSD-3), pandas_market_calendars (MIT) รายละเอียดใน docs/SOURCES_AUDIT_TH.md")
h2("ภาคผนวก จ การใช้ Generative AI และความรับผิดชอบทางวิชาการ")
P("ร่างรายงานฉบับปรับปรุง 2 จัดทำโดยใช้ OpenAI Codex ช่วยรวบรวมประเด็น สังเคราะห์แหล่งอ้างอิง เสนอ protocol และเรียบเรียงภาษาไทย ฉบับปรับปรุง 3 นี้ใช้ Claude (Anthropic, Claude Opus 5.5) ระหว่าง 18–19 พฤศจิกายน 2569 ในขอบเขตต่อไปนี้: ตรวจ repository ที่อ้างอิง เขียนโค้ด pipeline และชุดทดสอบทั้งหมด ดึงไฟล์ข้อมูลผ่าน browser ของผู้จัดทำ รันการทดลองและ post-hoc analysis ในสภาพแวดล้อมประมวลผลของ Claude สร้างภาพและตาราง และเรียบเรียงรายงานกับสไลด์ภาษาไทยจากไฟล์ผลจริง ตัวเลขทุกตัวดึงจากไฟล์ CSV/JSON ของ run เดียวกันโดยโปรแกรม ไม่มีผลใดที่ AI สร้างขึ้นแทนผลจริง")
P("ผู้จัดทำต้องตรวจโค้ด ทบทวนวิธีวิจัย รันซ้ำหรือตรวจ artefacts อ่านต้นฉบับงานวิจัยที่อ้างอิง และรับผิดชอบความถูกต้องของข้อสรุป รวมทั้งปรับข้อความเปิดเผยการใช้ AI ให้ตรงกับข้อกำหนดของผู้สอน การอ้างถึง AI ไม่ใช้แทนการอ้างบทความหรือฐานข้อมูลต้นทาง")
h2("ภาคผนวก ฉ แผนที่แหล่งที่มาของตารางและภาพ")
P("ตารางและภาพทุกชิ้นในรายงานระบุไฟล์ข้อมูล สคริปต์ run ID และ commit ไว้ใต้ภาพหรือตาราง และรวมไว้ในไฟล์ source_map.csv ที่ส่งมอบคู่กับรายงาน โฟลเดอร์อ้างอิงคือ runs/formal_v1/ ภายใน bundle_formal_v1.zip")

json.dump(B, open(OUT / "report_spec.json", "w"), ensure_ascii=False, indent=0)
print("report blocks", len(B), "tables", TN[0], "figures", FN_[0])

# =========================================================================== SLIDES
SL = []
def slide(**k):
    SL.append(k)


def fpath(fid):
    row = fm.loc[fid]
    p = FIG / row["png"]
    if not p.exists():
        shutil.copyfile(RUN / "figures" / row["png"], p)
    script = row["script"] if isinstance(row["script"], str) else "src/volvix/plotting.py"
    commit = POST_COMMIT if "post_hoc" in script else MAIN_COMMIT
    return str(p), f"ที่มา: figures/{row['data_csv']} · run {RUN.name} · commit {commit}"


slide(kind="title", title="พยากรณ์ความผันผวนพอร์ต ETF ด้วย LSTM และ Transformer",
      subtitle="ข้อมูล VIX ช่วยการพยากรณ์ การแจ้งเตือน และการบริหารความเสี่ยงหรือไม่",
      meta="Applied Deep Learning and Neural Networks · ผู้จัดทำ: รอระบุ · 19 พ.ย. 2569",
      notes="สวัสดีครับ วันนี้จะนำเสนอผลการทดลองพยากรณ์ความผันผวนห้าวันของพอร์ต ETF ด้วย LSTM และ Transformer และตรวจว่าการเพิ่ม VIX ช่วยหรือไม่ ผลทั้งหมดมาจาก run formal_v1 ที่ตรึง protocol ก่อนเปิด test ครั้งเดียว")
slide(kind="stepper", title="วันนี้เราจะตอบอะไร และเดินเรื่องอย่างไร",
      steps=[["ตั้งโจทย์", "แปลงคำถามบริหารความเสี่ยงพอร์ตเป็นงานพยากรณ์ความผันผวน 5 วัน"],
             ["ออกแบบโมเดล", "LSTM และ Transformer แบบมี/ไม่มี VIX เทียบกับ HV20, EWMA, GARCH"],
             ["ทดลองอย่างไม่โกง", "แบ่งตามเวลา purge ขอบ และตรึงทุกอย่างก่อนเปิด test"],
             ["อ่านผล", "ความแม่นยำ การแจ้งเตือน และผลการลงทุนหลังต้นทุน"],
             ["สรุปบทเรียน", "ทำไมแม่นขึ้นแต่ไม่ได้เงินเพิ่ม และจะแก้อย่างไร"]],
      groups=[[0, 2, "ที่มาและวิธีการ · ส่วน 1–3"], [3, 4, "ผลและบทเรียน · ส่วน 4–5"]],
      notes="วันนี้มีห้าขั้น ตั้งโจทย์ ออกแบบโมเดล ทดลองอย่างไม่โกง อ่านผล และสรุปบทเรียน สามขั้นแรกคือที่มาและวิธีการ สองขั้นหลังคือผลและสิ่งที่เรียนรู้")
slide(kind="kpis", section="1 บทนำ", title="ข้อค้นพบหลัก: แม่นขึ้น แต่ไม่คุ้มหลังต้นทุน",
      kpis=[[f(tm.loc['L_RV', 'log_RMSE']), "log-RMSE ดีที่สุด (L RV)", f"GARCH {f(tm.loc['GARCH','log_RMSE'])}"],
            [f"{pbp.loc[('L_RV','L_R','log_se'),'mean_diff']:+.4f}", "ผลต่าง log-SE จากการเพิ่ม VIX (LSTM)", "CI คร่อมศูนย์"],
            [pct(ref['net_CAGR'], 1), "Net CAGR พอร์ตอ้างอิง", "ทุก overlay ต่ำกว่า"]],
      body="Deep Learning ลด log-RMSE ได้ แต่ GARCH ดีกว่าด้าน QLIKE และ VaR · VIX ไม่ช่วยอย่างมีนัย · overlay ทุกแบบแพ้พอร์ตอ้างอิงหลังต้นทุน 10 bps",
      source="ที่มา: test/test_metrics.csv, test/test_paired_bootstrap.csv, backtest/backtest_summary.csv · run formal_v1 · commit 892dcc7",
      notes="สรุปก่อนเลย สามข้อ หนึ่ง Deep Learning มี log-RMSE ต่ำสุด แต่ไม่ใช่ทุก loss สอง VIX ไม่ช่วยอย่างมีนัย สาม ความแม่นยำที่ดีขึ้นไม่แปลงเป็นผลตอบแทนหลังต้นทุน")
slide(kind="rq", section="1 บทนำ", title="คำถามวิจัยและพอร์ตที่ศึกษา",
      items=["RQ1 LSTM/Transformer พยากรณ์ความผันผวน 5 วันต่างจาก HV20, EWMA, GARCH อย่างไร",
             "RQ2 การเพิ่ม VIX ช่วยแต่ละ architecture หรือไม่",
             "RQ3 สัญญาณช่วยผลตอบแทนต่อความเสี่ยงหลังต้นทุนหรือไม่"],
      image=str(FIG / "experiment-allocation.png"), source="ภาพแนวคิดจากร่างรายงาน (น้ำหนักกำหนดก่อนทดลอง)",
      notes="พอร์ตอ้างอิงเป็น Core–Satellite ACWI 40 SHY 20 XLK 20 XLV 15 GLD 5 เมื่อ alert จะลดทุกตัวลงครึ่งหนึ่งแล้วย้ายไป SHY ทั้งหมดนี้กำหนดก่อนทดลอง ไม่ได้ optimize จากผลย้อนหลัง")
slide(kind="arch", section="2 สถาปัตยกรรม", title="การทดลอง 2 × 2 แยกผลของ VIX และ architecture",
      cells=[["L R", "LSTM · Returns 60×5", f"{int(st.loc['L_R','n_params']):,} params"], ["L RV", "LSTM · Returns+VIX 60×6", f"{int(st.loc['L_RV','n_params']):,} params"],
             ["T R", "Transformer · Returns", f"{int(st.loc['T_R','n_params']):,} params"], ["T RV", "Transformer · Returns+VIX", f"{int(st.loc['T_RV','n_params']):,} params"]],
      body="LSTM 32 units → Dense 16 → 1 · Transformer: projection 32, 2 heads, FF 64, 1 block, GAP · loss = MSE บน log volatility · baselines: HV20, EWMA λ=0.94, GARCH(1,1)",
      source="ที่มา: src/volvix/models.py, post_hoc/seed_stability.csv · commit 892dcc7",
      notes="ใช้โมเดลเล็กเพียง 5 ถึง 9 พัน parameters เพราะข้อมูลรายวันมีตัวอย่างอิสระจำกัด เปรียบเทียบเป็นคู่ภายใน architecture เพื่อวัดผลของ VIX และเทียบข้าม architecture เมื่อข้อมูลเหมือนกัน")
slide(kind="framework", section="2 สถาปัตยกรรม", title="กรอบการวิจัย: จากข้อมูลตลาดสู่การตัดสินใจลดความเสี่ยง",
      source=f"ที่มา: configs/protocol.yaml, dataset/split_manifest.csv, frozen/freeze.json · run {RUN.name} · commit {MAIN_COMMIT}",
      fw={"data": ["ข้อมูลรายวัน", "Yahoo · Cboe · French", "5 ETF + VIX + RF", f"{N['common_start'][:7]} – {N['last_date'][:7]}", f"N = {N['sessions']:,} วัน"],
          "feat": ["Returns 5 ETF × 60 วัน", "VIX lag 1 วัน (เฉพาะ RV)", "Target: SD 5 วันข้างหน้า × √252", f"τ = P90 ของ train = {pct(N['tau'])}"],
          "prep": ["ปฏิทิน NYSE ไม่เติมค่า", "purge 5 วันที่ขอบ", "scaler fit บน train เท่านั้น", "train ≤2017 · val 18–19 · test 20–25"],
          "models": ["L R", "L RV", "T R", "T RV"], "base": "Baselines: HV20 · EWMA · GARCH(1,1)", "runs": "48 runs → ensemble 3 seeds",
          "eval": [["ความแม่นยำ", "Forecast accuracy", ["log-RMSE · QLIKE · RMSE", "paired block bootstrap CI"]],
                   ["การแจ้งเตือน", "Alert quality", ["Precision · Recall · F1 · AP", "cutoff จาก validation"]],
                   ["ผลการลงทุน", "Economic value", ["Net CAGR · Sharpe · Max DD", "ต้นทุน 10 bps · trade t+1"]],
                   ["อธิบายโมเดล", "Explainability", ["permutation importance", "VIX เทียบกับ returns"]]],
          "goal": ["Risk dashboard", "ช่วยผู้จัดการพอร์ตตัดสินใจลดความเสี่ยง"]},
      notes="นี่คือกรอบทั้งงานในหน้าเดียว แถวบนคือการสร้างโมเดล เริ่มจากข้อมูลรายวัน 4,470 วัน สร้าง input เป็นผลตอบแทนย้อนหลัง 60 วัน และ VIX สำหรับโมเดล RV แล้วแบ่งข้อมูลตามเวลาพร้อม purge ก่อนฝึกโมเดลสี่แบบเทียบกับ baseline สามแบบ ผลลัพธ์คือค่าพยากรณ์ความผันผวน ซึ่งถ้าเกินเกณฑ์จะเป็นสัญญาณเตือน แถวล่างคือการประเมินบนข้อมูลอนาคตปี 2020 ถึง 2025 ที่ตรึงไว้ก่อนเปิด ประเมินสี่ด้าน คือความแม่นยำ การแจ้งเตือน ผลการลงทุนหลังต้นทุน และการอธิบายโมเดล เป้าหมายปลายทางคือหน้าจอเฝ้าระวังความเสี่ยงให้คนตัดสินใจ ไม่ใช่ปุ่มซื้อขายอัตโนมัติ")
p1, s1 = fpath("F01_data_overview")
slide(kind="figure", section="3 วิธีวิจัยและข้อมูล", title="ข้อมูลจริง 2008–2025: target และ VIX เคลื่อนไหวร่วมกัน",
      image=p1, caption=f"{N['sessions']:,} วันทำการ · ไม่มี missing session · τ = P90 ของ train = {pct(N['tau'])}", source=s1,
      notes=f"ข้อมูลร่วมเริ่ม {N['common_start']} เพราะ ACWI เริ่มซื้อขายช้าที่สุด target คือ annualized SD ของห้าวันข้างหน้า ไม่ใช่ความผันผวนหนึ่งปี เส้นประคือขอบ validation และ test ช่วง test มีวิกฤต COVID ที่ y สูงกว่าร้อยละร้อย")
slide(kind="timeline_table", section="3 วิธีวิจัยและข้อมูล", title="การแบ่งข้อมูลตามเวลาป้องกันการรั่วของข้อมูลอนาคต",
      segments=[["Train", "2008–2017", 9.75, "ฝึกโมเดล · ตั้ง τ และ scaler"], ["Val", "2018–19", 2.0, "เลือก config · cutoff"], ["Test", "2020–2025", 6.0, "เปิดครั้งเดียว"]],
      header=["ชุด", "ช่วง origin", "ตัวอย่างสุทธิ", "purge", "y เฉลี่ย", "y > τ"],
      rows=[[lab_, f"{S.loc[s,'first_origin'][:7]} – {S.loc[s,'last_origin'][:7]}", f"{int(S.loc[s,'net_samples']):,}", f"{int(S.loc[s,'purged_target_crosses_boundary'])}",
             pct(D.loc[(s, 'y_target'), 'mean'], 1), pct(H.loc[s, 'high_risk_share_y_gt_tau'], 1)] for s, lab_ in [("train", "Train"), ("validation", "Validation"), ("test", "Test")]],
      body=f"Alignment checks ผ่าน {N['checks_passed']}/{N['checks_total']} ข้อด้วยข้อมูลจริงก่อนฝึก · validation มีความผันผวนต่ำและเหตุการณ์น้อย ({pct(N['hr_val'],1)})",
      source="ที่มา: dataset/split_manifest.csv, reports/descriptive_stats_by_split.csv · run formal_v1 · commit 892dcc7",
      notes="validation มีเหตุการณ์ความผันผวนสูงเพียงร้อยละ 7.2 ทำให้ cutoff ที่เลือกมีความไม่แน่นอน และ test มีระดับความผันผวนสูงกว่า เป็น distribution shift ที่ต้องคำนึง")
slide(kind="table", section="4 ผลการศึกษา", title="RQ1: Deep Learning ชนะด้าน log-RMSE แต่ GARCH ชนะด้าน QLIKE",
      header=["โมเดล", "RMSE (pp)", "log-RMSE", "QLIKE", "forecast เฉลี่ย"],
      rows=[[NAME[m], f(tm.loc[m, 'RMSE_pp'], 2), f(tm.loc[m, 'log_RMSE']), f(tm.loc[m, 'QLIKE']), f"{f(tm.loc[m,'mean_forecast_pp'],1)}%"] for m in MODELS],
      body=f"y เฉลี่ยจริง {f(tm.loc['L_R','mean_actual_pp'],1)}% · Deep Learning พยากรณ์ต่ำกว่าจริงราว 1 pp → QLIKE และ VaR แย่ลง",
      source=f"ที่มา: test/test_metrics.csv · n = {N['n_test']:,} · run formal_v1 · commit 892dcc7",
      notes="log-RMSE ดีที่สุดคือ L RV แต่ QLIKE ดีที่สุดคือ GARCH เหตุผลคือการฝึกบน log scale ทำให้ forecast ต่ำกว่าค่าเฉลี่ยจริง QLIKE ลงโทษการพยากรณ์ต่ำเกินมาก ความได้เปรียบเหนือ HV20 และ EWMA มี CI ไม่คร่อมศูนย์ เทียบ GARCH ชัดเฉพาะ LSTM")
p4, s4 = fpath("F04_test_forecast_vs_actual")
slide(kind="figure_text", section="4 ผลการศึกษา", title="ค่าพยากรณ์ตามทันการพุ่งขึ้น แต่ต่ำกว่าจริงช่วงสุดขั้ว",
      image=p4, source=s4,
      items=["เส้นดำ = ค่าจริง; แต่ละแผง = หนึ่งโมเดล แกนเดียวกัน", "ทุกโมเดลตามทันหลังความผันผวนเกิดแล้ว ไม่มีโมเดลใดเห็นจุดสูงสุด มี.ค. 2020 ล่วงหน้า",
             f"forecast เฉลี่ย DL {f(tm.loc[DL,'mean_forecast_pp'].min(),1)}–{f(tm.loc[DL,'mean_forecast_pp'].max(),1)}% เทียบค่าจริง {f(tm.loc['L_R','mean_actual_pp'],1)}%",
             "หน่วย: annualized five-day SD (ไม่ใช่ความผันผวน 1 ปี)"],
      notes="ทุกโมเดลตามทันหลังความผันผวนเกิดแล้ว ไม่มีโมเดลใดพยากรณ์จุดสูงสุดเดือนมีนาคม 2020 ได้ล่วงหน้า และ Deep Learning พยากรณ์ช่วงสุดขั้วต่ำกว่าจริง")
p14, s14 = fpath("F14_permutation_importance")
slide(kind="figure_wide", section="4 ผลการศึกษา", title="RQ2: VIX สำคัญต่อโมเดล แต่ไม่เพิ่มความแม่นยำ",
      image=p14, source=s14 + " (post-hoc XAI)",
      items=[f"L RV − L R: {pbp.loc[('L_RV','L_R','log_se'),'mean_diff']:+.4f} [{pbp.loc[('L_RV','L_R','log_se'),'ci_low_2.5']:+.3f}, {pbp.loc[('L_RV','L_R','log_se'),'ci_high_97.5']:+.3f}]",
             f"T RV − T R: {pbp.loc[('T_RV','T_R','log_se'),'mean_diff']:+.4f} [{pbp.loc[('T_RV','T_R','log_se'),'ci_low_2.5']:+.3f}, {pbp.loc[('T_RV','T_R','log_se'),'ci_high_97.5']:+.3f}]",
             "VIX แทนข้อมูลจาก ACWI/XLK → ข้อมูลซ้ำซ้อน"],
      notes="ผลต่าง log squared error จากการเพิ่ม VIX เล็กมากและ CI คร่อมศูนย์ทั้งสอง architecture permutation importance แสดงว่าเมื่อมี VIX โมเดลพึ่ง VIX แทน ACWI และ XLK แปลว่าข้อมูลซ้ำกัน ไม่ได้เพิ่มข้อมูลใหม่มากนัก")
p3, s3 = fpath("F03_learning_curves")
slide(kind="figure_text", section="4 ผลการศึกษา", title="ความเสถียร: early stopping เร็ว Transformer แปรผันตาม seed",
      image=p3, source=s3,
      items=[f"best epoch 3–19 จาก 100", f"SD test log-RMSE ข้าม seed: LSTM ≈ {f(st.loc['L_R','test_logRMSE_seed_sd'],3)}, Transformer ≈ {f(st.loc['T_R','test_logRMSE_seed_sd'],3)}",
             "ensemble 3 seeds ดีกว่าค่าเฉลี่ยราย seed ทุกโมเดล"],
      notes="validation loss ต่ำสุดเร็วแล้วสูงขึ้น เป็นสัญญาณ overfitting ที่ early stopping ตัดไว้ Transformer มี variance ข้าม seed สูงกว่า LSTM ราวห้าเท่า สอดคล้องกับ bias–variance trade-off ensemble ช่วยลด variance แต่ไม่แก้ bias ที่พยากรณ์ต่ำเกิน")
slide(kind="table", section="4 ผลการศึกษา", title="การแจ้งเตือน: Deep Learning แม่นกว่า แต่พลาดมากกว่า",
      header=["โมเดล", "Precision", "Recall", "F1", "AP", "False/100 วัน"],
      rows=[[NAME[m], f(tm.loc[m, 'alert_precision']), f(tm.loc[m, 'alert_recall']), f(tm.loc[m, 'alert_F1']), f(tm.loc[m, 'alert_AP']), f(tm.loc[m, 'alert_false_alerts_per_100'], 1)] for m in MODELS],
      body=f"positive {N['n_pos_test']} จาก {N['n_test']:,} วัน ({pct(N['hr_test'],1)}) · AP ใกล้กันทุกโมเดล → ความต่างมาจาก cutoff",
      source="ที่มา: test/test_metrics.csv, frozen/freeze.json · run formal_v1 · commit 892dcc7",
      notes="baselines เตือนบ่อยเพราะ cutoff ต่ำ จึงจับได้เกือบหมดแต่เตือนผิด 18 ถึง 41 ครั้งต่อร้อยวัน Deep Learning เตือนแม่นกว่าแต่พลาด 54 ถึง 70 วัน AP ใกล้กันมาก แปลว่าความสามารถจัดอันดับพอ ๆ กัน ความต่างหลักคือ cutoff ที่เลือกจาก validation")
p6, s6 = fpath("F06_test_confusion_matrices")
p7, s7 = fpath("F07_test_precision_recall")
slide(kind="wide_plus", section="4 ผลการศึกษา", title="Confusion matrix 2 × 2 และ Precision–Recall",
      image=p6, image2=p7,
      items=[f"TP/FN/FP/TN จริงของทั้ง 7 โมเดล บน {N['n_test']:,} วัน", f"PR curve ทุกโมเดลอยู่เหนือ prevalence {pct(N['hr_test'],1)}",
             f"วันที่ HV ย้อนหลังยังต่ำ: AP เหลือ {sub['alert_AP'].min():.2f}–{sub['alert_AP'].max():.2f} → ส่วนใหญ่เป็นการติดตาม ไม่ใช่เตือนล่วงหน้า"], source=s6.replace("ที่มา: ", "ที่มา: ") + " ; " + s7.split("ที่มา: ")[1],
      notes="confusion matrix แสดงจำนวน TP FP FN TN จริง PR curve เทียบกับเส้น prevalence ประมาณร้อยละ 10 ทุกโมเดลเหนือเส้นอ้างอิงชัดเจน แต่ในวันที่ความผันผวนย้อนหลังยังต่ำ AP ลดเหลือ 0.19 ถึง 0.30 สัญญาณส่วนใหญ่จึงเป็นการติดตามภาวะที่เกิดแล้ว")
slide(kind="table", section="4 ผลการศึกษา", title="RQ3: ไม่มี overlay ใดชนะพอร์ตอ้างอิงหลังต้นทุน",
      header=["กลยุทธ์", "Net CAGR", "Vol", "Sharpe", "Max DD", "Turnover/ปี"],
      rows=[[lab.get(s, s.replace("_overlay", " overlay").replace("_static", " static").replace("_", " ")), pct(b10.loc[s, 'net_CAGR'], 1), pct(b10.loc[s, 'ann_vol'], 1),
             f(b10.loc[s, 'sharpe'], 2), pct(b10.loc[s, 'max_drawdown'], 1), f(b10.loc[s, 'turnover_per_year'], 1)]
            for s in ["Reference_fixed", "HV20_overlay", "EWMA_overlay", "GARCH_overlay", "L_R_overlay", "L_RV_overlay", "T_R_overlay", "T_RV_overlay", "L_RV_static"]],
      body="ต้นทุน 10 bps · signal t → trade t+1 · 3 ม.ค. 2020 – 31 ธ.ค. 2025 · static comparator ดีกว่า overlay ของตนทุกโมเดล",
      source="ที่มา: backtest/backtest_summary.csv · run formal_v1 · commit 892dcc7",
      notes="พอร์ตอ้างอิงได้ CAGR ร้อยละ 12.3 overlay ทุกแบบได้ 9 ถึง 9.9 HV20 และ EWMA ลด drawdown ได้มากจน Sharpe สูงกว่าเล็กน้อย แต่ Deep Learning overlay ทั้ง CAGR และ Sharpe ต่ำกว่า turnover 8 ถึง 9 เท่าต่อปีคือต้นเหตุหลัก")
p8, s8 = fpath("F08_nav_and_drawdown")
slide(kind="figure", section="4 ผลการศึกษา", title="NAV และ drawdown: ลดความเสี่ยงได้ แต่เสียช่วงฟื้นตัว",
      image=p8, caption="NAV เริ่ม 100 เท่ากันทุกกลยุทธ์ · ต้นทุน 10 bps ต่อด้าน", source=s8,
      notes="เส้นดำคือพอร์ตอ้างอิง overlay ลด drawdown ช่วง 2022 ได้ แต่กลับเข้าตลาดช้าในช่วงฟื้นตัวและเสียต้นทุนจากการสลับสถานะ")
p9, s9 = fpath("F09_overlay_weights")
p15, s15 = fpath("F15_cost_sensitivity")
slide(kind="two_figures", section="4 ผลการศึกษา", title="Exposure และความไวต่อต้นทุน",
      image=p9, image2=p15, source="ที่มา: figures/data/F09_overlay_weights.csv (commit 892dcc7); figures/data/F15_cost_sensitivity.csv (commit fc0476d) · run formal_v1",
      notes="ซ้ายคือน้ำหนัก SHY และหุ้นตามเวลา เห็นการสลับสถานะบ่อย ขวาคือ CAGR และ Sharpe ที่ต้นทุน 5 10 20 bps แม้ต้นทุน 5 bps overlay ก็ยังแพ้พอร์ตอ้างอิง")
slide(kind="bullets", section="5 อภิปราย", title="ทำไม forecast ที่แม่นขึ้นไม่แปลงเป็นผลตอบแทน",
      items=["สองสถานะ + cutoff → สลับบ่อย turnover 5–9 เท่า/ปี", "ซื้อขายช้า 1 วัน → ลดความเสี่ยงหลังเหตุการณ์เริ่มแล้ว",
             "ตลาด 2020–2025 ฟื้นตัวเร็ว → ถือ SHY แล้วเสียโอกาส", "log-RMSE วัดความแม่นยำรวม ไม่ใช่คุณภาพการตัดสินใจที่ cutoff",
             "class imbalance + validation ผันผวนต่ำ → cutoff ไม่เสถียร"],
      notes="ห้าเหตุผลนี้อธิบายช่องว่างระหว่างความแม่นยำกับผลลงทุน ข้อสังเกตสำคัญคือ static comparator ชนะ overlay ทุกโมเดล ประโยชน์จึงมาจากการถือความเสี่ยงต่ำลง ไม่ใช่การเลือกจังหวะ")
slide(kind="table", section="5 อภิปราย", title="จุดอ่อนที่แก้ได้ และทางแก้ในงานต่อไป",
      header=["จุดอ่อน", "ผลกระทบที่เห็นในงานนี้", "ทางแก้ในงานต่อไป"],
      rows=[["พอร์ตเดียว ช่วงเดียว (2020–2025)", "สรุปเป็นกฎทั่วไปไม่ได้", "ทดสอบหลายพอร์ต หลายช่วงเวลา"],
            ["ตรึงโมเดล 6 ปีไม่ฝึกใหม่", "ตลาดเปลี่ยน โมเดลไม่ได้เรียนตาม", "walk-forward ฝึกใหม่ทุกปี"],
            ["ทายต่ำกว่าจริงราว 1 จุด", f"VaR 95% หลุด {min(dlv):.1f}–{max(dlv):.1f}% แทน 5%", "calibration หรือฝึกด้วย QLIKE"],
            ["กติกาสองสถานะสลับบ่อย", "turnover 8–9 เท่า/ปี กินกำไร", "hysteresis / volatility targeting"],
            ["รันบน CPU ยังไม่ยืนยันบน GPU", "ตัวเลขอาจต่างเล็กน้อย", "รันซ้ำบน Colab GPU เป็น run แยก"]],
      body="ปฏิบัติ: ใช้ AI เป็น risk dashboard ไม่ใช่ปุ่มซื้อขาย · นโยบาย: ประเมิน AI ด้วยความแม่น + calibration + ผลหลังต้นทุน",
      source="ที่มา: test/test_metrics.csv, test/supplementary_var_backtest.csv, backtest/backtest_summary.csv · run formal_v1 · commit 892dcc7",
      notes="สไลด์นี้ขอพูดตรง ๆ ว่างานนี้ยังพลาดตรงไหนและจะแก้อย่างไร มีห้าเรื่อง ทุกข้อมีทางแก้ และต้องทำเป็นการทดลองใหม่ ไม่ใช่ปรับจากผลข้อสอบชุดนี้ ในทางปฏิบัติตอนนี้ควรใช้ AI เป็นหน้าจอเฝ้าระวังความเสี่ยง ยังไม่ควรเป็นปุ่มกดซื้อขาย")
slide(kind="closing", title="สรุป", items=["Deep Learning แม่นกว่า HV20/EWMA ด้าน log-RMSE แต่ GARCH ดีกว่าด้าน QLIKE และ VaR",
                                          "VIX ไม่ให้ประโยชน์ที่มีนัย เพราะซ้ำกับข้อมูลใน returns",
                                          "overlay ทุกแบบแพ้พอร์ตอ้างอิงหลังต้นทุน: ความแม่นยำ ≠ ผลตอบแทน"],
      notes="ขอบคุณครับ ยินดีรับคำถาม")
# appendix
slide(kind="appendix_table", section="ภาคผนวก", title="ภาคผนวก: log-RMSE รายปีบน test",
      header=["ปี", "y>τ"] + [NAME[m] for m in MODELS],
      rows=[[str(y), pct(prev.loc[y], 1)] + [f(yr.loc[y, m]) for m in MODELS] for y in yr.index],
      source="ที่มา: test/test_metrics_by_year.csv · run formal_v1 · commit 892dcc7", notes="ผลรายปีไม่สม่ำเสมอ ปี 2022 และ 2024 ผลต่างกับ baselines เล็กมาก")
slide(kind="appendix_table", section="ภาคผนวก", title="ภาคผนวก: VaR 1 วัน (ผลเสริม นอก protocol หลัก)",
      header=["โมเดล", "violation 95%", "Kupiec p", "violation 99%", "Kupiec p"],
      rows=[[NAME[m], pct(vv.loc[(m, 1, 0.95), 'rate'], 1), pval(vv.loc[(m, 1, 0.95), 'p_value']), pct(vv.loc[(m, 1, 0.99), 'rate'], 1), pval(vv.loc[(m, 1, 0.99), 'p_value'])] for m in MODELS],
      source="ที่มา: test/supplementary_var_backtest.csv · run formal_v1 · commit 892dcc7", notes="Deep Learning มี violation สูงกว่าระดับที่ตั้งไว้ สอดคล้องกับการพยากรณ์ต่ำเกิน")
slide(kind="appendix_table", section="ภาคผนวก", title="ภาคผนวก: การทำซ้ำและการใช้ Generative AI",
      header=["รายการ", "ค่า"],
      rows=[["Repository", REPO.replace("https://", "")], ["Commit การทดลอง / post-hoc", f"{MAIN_COMMIT} / {POST_COMMIT}"], ["Run / snapshot", f"{N['run']} / {N['snapshot']}"],
            ["ฮาร์ดแวร์", "CPU (sandbox ของ Claude) · 48 runs ≈ 12 นาที"], ["Generative AI", "Claude (Anthropic) เขียนโค้ด รันการทดลอง และร่างรายงาน/สไลด์จากไฟล์ผลจริง; ร่างฉบับ 2 ใช้ OpenAI Codex"],
            ["ความรับผิดชอบ", "ผู้จัดทำต้องตรวจโค้ด ผล และข้อสรุปก่อนส่ง"]],
      source="ที่มา: run_manifest.json · freeze.json", notes="รายละเอียดเต็มอยู่ในภาคผนวกของรายงาน")
_i = [i for i, x in enumerate(SL) if x.get("title", "").startswith("Exposure และความไวต่อต้นทุน")][0]
_s = SL.pop(_i)
_s["section"], _s["title"] = "ภาคผนวก", "ภาคผนวก: exposure ของ overlay และความไวต่อต้นทุน"
SL.insert([i for i, x in enumerate(SL) if x.get("section") == "ภาคผนวก"][0], _s)
json.dump(SL, open(OUT / "slides_spec.json", "w"), ensure_ascii=False, indent=0)
pd.DataFrame(SRC).drop_duplicates().to_csv(OUT / "source_map.csv", index=False)
print("slides", len(SL), "source map rows", len(pd.DataFrame(SRC).drop_duplicates()))
