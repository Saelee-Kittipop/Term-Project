// Render slides_spec.json into a Thai 16:9 PPTX (Sarabun) with real figures and Thai speaker notes.
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/root/.claude/skills/synced/60dea9bc-2fc2-4ff8-b28a-65021759eedb_f1494778-6e92-44ca-bc4a-00570146328e/pptx/scripts/apply_theme.js");

const SL = JSON.parse(fs.readFileSync("/home/claude/deliver/slides_spec.json", "utf8"));
const THEME = {
  name: "VolVIX Thai", headFontFace: "Sarabun", bodyFontFace: "Sarabun",
  colors: { dk1: "1B2333", lt1: "FFFFFF", dk2: "1F3864", lt2: "F2F4F8", accent1: "1F3864", accent2: "C8553D",
    accent3: "2E86AB", accent4: "7A8699", accent5: "E9B44C", accent6: "3B8E5A", hlink: "2E86AB", folHlink: "7A8699" },
};
const HEX = THEME.colors;
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "พยากรณ์ความผันผวนพอร์ต ETF ด้วย LSTM และ Transformer";
const C = pres.SchemeColor;
const W = 13.333, H = 7.5, MX = 0.55;

pres.defineSlideMaster({ title: "DARK", background: { color: HEX.dk2 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: MX, y: 1.95, w: W - 2 * MX, h: 1.7, fontSize: 36, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } }] });
pres.defineSlideMaster({ title: "CONTENT", background: { color: HEX.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.55, w: W - 2 * MX, h: 0.8, fontSize: 26, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { line: { x: MX, y: 1.42, w: 1.1, h: 0, line: { color: HEX.accent2, width: 3 } } },
  ],
  slideNumber: { x: W - 1.0, y: H - 0.42, w: 0.6, h: 0.3, fontSize: 10, color: HEX.accent4, align: "right" } });

function png(path) { const b = fs.readFileSync(path); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function fit(path, x, y, w, h) {
  const s = png(path), r = s.w / s.h;
  let ww = w, hh = w / r; if (hh > h) { hh = h; ww = h * r; }
  return { path, x: x + (w - ww) / 2, y: y + (h - hh) / 2, w: ww, h: hh };
}
const T = (s, text, o) => s.addText(text, { isTextBox: true, fontFace: "Sarabun", color: HEX.dk1, margin: 0, ...o });
function section(s, txt) { if (txt) T(s, txt.toUpperCase(), { x: MX, y: 0.22, w: 8, h: 0.3, fontSize: 12, color: HEX.accent2, bold: true, objectName: "section" }); }
function source(s, txt) { if (txt) T(s, txt, { x: MX, y: H - 0.48, w: W - 2 * MX - 0.8, h: 0.34, fontSize: 9.5, color: HEX.accent4, valign: "middle", objectName: "source" }); }
function bullets(s, items, x, y, w, h, fs = 18) {
  T(s, items.map((t, i) => ({ text: t, options: { bullet: { indent: 18 }, breakLine: i < items.length - 1, paraSpaceAfter: 10 } })),
    { x, y, w, h, fontSize: fs, valign: "top", objectName: "bullets" });
}
function table(s, header, rows, x, y, w, fs = 13) {
  const hdr = header.map((t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, align: "center" } }));
  const body = rows.map((r, ri) => r.map((t, i) => ({ text: String(t), options: { align: i === 0 ? "left" : "center", fill: { color: ri % 2 ? HEX.lt2 : HEX.lt1 } } })));
  s.addTable([hdr, ...body], { x, y, w, fontFace: "Sarabun", fontSize: fs, color: HEX.dk1, border: { type: "solid", pt: 0.5, color: "C9CFD8" },
    rowH: Math.min(0.42, 4.6 / (rows.length + 1)), valign: "middle", margin: [2, 6, 2, 6], objectName: "table" });
}

let sec = null;
for (const d of SL) {
  const secTitle = d.kind === "title" || d.kind === "agenda" || d.kind === "stepper" ? "เปิด" : d.kind === "closing" ? "สรุป" : (d.section || "").split(" ").slice(0, 2).join(" ") || "เปิด";
  const secName = d.section === "ภาคผนวก" ? "ภาคผนวก" : secTitle;
  if (secName !== sec) { pres.addSection({ title: secName }); sec = secName; }
  const dark = d.kind === "title" || d.kind === "closing";
  const s = pres.addSlide({ masterName: dark ? "DARK" : "CONTENT", sectionTitle: secName });
  s.addText(d.title, { placeholder: "title" });
  if (!dark) section(s, d.section);
  const top = 1.7, bottom = H - 0.6, avail = bottom - top;
  switch (d.kind) {
    case "title":
      T(s, d.subtitle, { x: MX, y: 3.95, w: W - 2 * MX, h: 0.7, fontSize: 22, color: "CADCFC" });
      T(s, d.meta, { x: MX, y: 5.2, w: W - 2 * MX, h: 0.5, fontSize: 14, color: "CADCFC" });
      s.addShape(pres.shapes.LINE, { x: MX, y: 3.88, w: 1.4, h: 0, line: { color: HEX.accent5, width: 3 } });
      break;
    case "closing":
      s.addShape(pres.shapes.LINE, { x: MX, y: 3.88, w: 1.4, h: 0, line: { color: HEX.accent5, width: 3 } });
      T(s, d.items.map((t, i) => ({ text: `${i + 1}  ${t}`, options: { breakLine: i < d.items.length - 1, paraSpaceAfter: 12 } })),
        { x: MX, y: 4.05, w: W - 2 * MX, h: 2.6, fontSize: 20, color: C.background1, valign: "top" });
      break;
    case "agenda":
      d.items.forEach((t, i) => {
        const y = top + 0.15 + i * 0.95;
        s.addShape(pres.shapes.OVAL, { x: MX, y, w: 0.62, h: 0.62, fill: { color: HEX.dk2 }, objectName: `num${i}` });
        T(s, String(i + 1), { x: MX, y, w: 0.62, h: 0.62, fontSize: 20, bold: true, color: HEX.lt1, align: "center", valign: "middle" });
        T(s, t.replace(/^\d\s/, ""), { x: MX + 0.9, y, w: 9, h: 0.62, fontSize: 22, valign: "middle" });
      });
      break;
    case "kpis": {
      const cw = (W - 2 * MX - 0.6) / 3;
      d.kpis.forEach(([v, l, n], i) => {
        const x = MX + i * (cw + 0.3);
        s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: top, w: cw, h: 2.6, rectRadius: 0.08, fill: { color: HEX.lt2 }, line: { color: HEX.lt2 }, objectName: `kpi${i}` });
        T(s, v, { x: x + 0.3, y: top + 0.25, w: cw - 0.6, h: 1.1, fontSize: 48, bold: true, color: i === 2 ? HEX.accent2 : HEX.dk2 });
        T(s, l, { x: x + 0.3, y: top + 1.4, w: cw - 0.6, h: 0.6, fontSize: 16, bold: true });
        T(s, n, { x: x + 0.3, y: top + 2.0, w: cw - 0.6, h: 0.45, fontSize: 14, color: HEX.accent4 });
      });
      T(s, d.body, { x: MX, y: top + 3.0, w: W - 2 * MX, h: 1.4, fontSize: 18, valign: "top" });
      break;
    }
    case "rq": case "text_image":
      bullets(s, d.items, MX, top, 5.6, avail, 18);
      s.addImage(fit(d.image, 6.5, top, W - 6.5 - MX, avail - 0.1));
      break;
    case "arch": {
      const cw = 2.9, ch = 1.35;
      d.cells.forEach(([code, desc, p], i) => {
        const x = MX + (i % 2) * (cw + 0.2), y = top + Math.floor(i / 2) * (ch + 0.2);
        const vix = code.endsWith("RV");
        s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h: ch, rectRadius: 0.08, fill: { color: vix ? HEX.dk2 : HEX.lt2 }, line: { color: HEX.dk2, width: 1 }, objectName: `cell${i}` });
        T(s, code, { x: x + 0.2, y: y + 0.12, w: cw - 0.4, h: 0.5, fontSize: 24, bold: true, color: vix ? HEX.lt1 : HEX.dk2 });
        T(s, desc, { x: x + 0.2, y: y + 0.62, w: cw - 0.4, h: 0.35, fontSize: 13, color: vix ? HEX.lt1 : HEX.dk1 });
        T(s, p, { x: x + 0.2, y: y + 0.95, w: cw - 0.4, h: 0.3, fontSize: 12, color: vix ? "CADCFC" : HEX.accent4 });
      });
      T(s, "แถว = architecture · คอลัมน์ = มี/ไม่มี VIX", { x: MX, y: top + 2 * ch + 0.5, w: 2 * cw + 0.2, h: 0.4, fontSize: 13, color: HEX.accent4, italic: true });
      bullets(s, d.body.split(" · "), 7.0, top, W - 7.0 - MX, avail, 16);
      break;
    }
    case "figure": {
      const capH = 0.45;
      s.addImage(fit(d.image, MX, top, W - 2 * MX, avail - capH - 0.05));
      T(s, d.caption, { x: MX, y: bottom - capH, w: W - 2 * MX, h: capH, fontSize: 14, color: HEX.dk1, align: "center", italic: true });
      break;
    }
    case "figure_text":
      s.addImage(fit(d.image, MX, top, 7.6, avail));
      bullets(s, d.items, 8.45, top + 0.1, W - 8.45 - MX, avail, 16);
      break;
    case "stepper": {
      const n = d.steps.length, x0 = MX + 0.9, x1 = W - MX - 0.9, gap = (x1 - x0) / (n - 1), cy = 3.55, r = 0.42;
      s.addShape(pres.shapes.LINE, { x: x0, y: cy, w: x1 - x0, h: 0, line: { color: "9DB7D5", width: 1.25 }, objectName: "rail" });
      d.groups.forEach(([a, b, lab], gi) => {
        const xa = x0 + a * gap - 1.0, xb = x0 + b * gap + 1.0;
        s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: xa, y: 2.45, w: xb - xa, h: 0.4, rectRadius: 0.2, fill: { color: "DCE8F5" }, line: { color: "DCE8F5" }, objectName: `group${gi}` });
        T(s, lab, { x: xa, y: 2.45, w: xb - xa, h: 0.4, fontSize: 13, bold: true, color: "2E5A88", align: "center", valign: "middle" });
      });
      d.steps.forEach(([t, desc], i) => {
        const cx = x0 + i * gap, last = i === n - 1;
        s.addShape(pres.shapes.OVAL, { x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, fill: { color: last ? "2E5A88" : HEX.lt1 }, line: { color: "2E5A88", width: 2 }, objectName: `step${i}` });
        T(s, String(i + 1), { x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, fontSize: 22, bold: true, color: last ? HEX.lt1 : "2E5A88", align: "center", valign: "middle" });
        T(s, t, { x: cx - 1.15, y: cy + 0.6, w: 2.3, h: 0.45, fontSize: 19, bold: true, align: "center" });
        T(s, desc, { x: cx - 1.15, y: cy + 1.1, w: 2.3, h: 1.5, fontSize: 15, color: HEX.accent4, align: "center", valign: "top" });
      });
      break;
    }
    case "framework": {
      const f = d.fw, NAVY = HEX.dk2, LINEC = "8A96A8";
      const bar = (txt, y) => { s.addShape(pres.shapes.RECTANGLE, { x: MX, y, w: 5.6, h: 0.34, fill: { color: NAVY }, line: { color: NAVY }, objectName: "bar" });
        T(s, txt, { x: MX + 0.15, y, w: 5.4, h: 0.34, fontSize: 12, bold: true, color: HEX.lt1, valign: "middle", charSpacing: 2 }); };
      const arrow = (x, y) => s.addShape(pres.shapes.RIGHT_ARROW, { x, y, w: 0.26, h: 0.26, fill: { color: HEX.dk1 }, line: { color: HEX.dk1 }, objectName: "arrow" });
      const box = (x, y, w, h, o = {}) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: o.fill || "F4F6F9" }, line: { color: o.line || LINEC, width: 1, dashType: o.dash }, objectName: o.name || "box" });
      const bul = (items, x, y, w, h, fs = 11.5) => T(s, items.map((t, i) => ({ text: t, options: { bullet: { code: "25CB" }, breakLine: i < items.length - 1, paraSpaceAfter: 4 } })), { x, y, w, h, fontSize: fs, valign: "top" });
      bar("MODEL DEVELOPMENT PIPELINE", 1.6);
      const top = 2.1, hh = 2.15, yA = top + hh / 2 - 0.13;
      // 1 data
      let x = MX;
      T(s, f.data[0], { x, y: top, w: 1.75, h: 0.35, fontSize: 14, bold: true, align: "center" });
      T(s, f.data.slice(1).join("\n"), { x, y: top + 0.42, w: 1.75, h: 1.7, fontSize: 11.5, align: "center", valign: "top", color: HEX.dk1 });
      arrow(x + 1.8, yA); x += 2.12;
      // 2 features
      T(s, "Input และ target", { x, y: top - 0.02, w: 2.55, h: 0.32, fontSize: 13, bold: true, align: "center" });
      box(x, top + 0.32, 2.55, hh - 0.32); bul(f.feat, x + 0.1, top + 0.45, 2.4, hh - 0.5);
      arrow(x + 2.6, yA); x += 2.92;
      // 3 prep
      T(s, "เตรียมข้อมูลและแบ่งตามเวลา", { x, y: top - 0.02, w: 2.45, h: 0.32, fontSize: 13, bold: true, align: "center" });
      box(x, top + 0.32, 2.45, hh - 0.32, { fill: "FBF3D5", line: "C9A227" }); bul(f.prep, x + 0.1, top + 0.45, 2.3, hh - 0.5);
      arrow(x + 2.5, yA); x += 2.82;
      // 4 models
      T(s, "พัฒนาโมเดล", { x, y: top - 0.02, w: 2.6, h: 0.32, fontSize: 13, bold: true, align: "center" });
      box(x, top + 0.32, 2.6, hh - 0.32);
      f.models.forEach((m, i) => { const bx = x + 0.15 + (i % 2) * 1.17, by = top + 0.45 + Math.floor(i / 2) * 0.5, vix = m.endsWith("RV");
        s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y: by, w: 1.1, h: 0.42, rectRadius: 0.05, fill: { color: vix ? NAVY : "E3EAF3" }, line: { color: NAVY, width: 0.75 }, objectName: `m${i}` });
        T(s, m, { x: bx, y: by, w: 1.1, h: 0.42, fontSize: 12, bold: true, color: vix ? HEX.lt1 : NAVY, align: "center", valign: "middle" }); });
      T(s, f.base, { x: x + 0.1, y: top + 1.47, w: 2.4, h: 0.3, fontSize: 10.5, align: "center" });
      T(s, f.runs, { x: x + 0.1, y: top + 1.75, w: 2.4, h: 0.3, fontSize: 10.5, color: HEX.accent4, align: "center" });
      arrow(x + 2.65, yA); x += 2.97;
      // 5 output
      const ow = W - MX - x;
      box(x, top + 0.32, ow, hh - 0.32, { fill: "FFFBF2", line: "8A96A8", dash: "dash" });
      T(s, [{ text: "Alert", options: { bold: true, color: "C0392B", breakLine: true } }, { text: "forecast > cutoff", options: { fontSize: 10, color: "C0392B", breakLine: true } },
             { text: " ", options: { breakLine: true, fontSize: 6 } }, { text: "Normal", options: { bold: true, color: "2E7D4F", breakLine: true } }, { text: "ถือพอร์ตปกติ", options: { fontSize: 10, color: "2E7D4F" } }],
        { x: x + 0.05, y: top + 0.45, w: ow - 0.1, h: hh - 0.55, fontSize: 13, align: "center", valign: "middle" });
      T(s, "Forecast vol 5 วัน", { x: x - 0.1, y: top - 0.02, w: ow + 0.2, h: 0.32, fontSize: 12, bold: true, align: "center" });
      // evaluation row
      bar("MODEL EVALUATION · OUT-OF-TIME TEST 2020–2025", 4.5);
      const ey = 4.98, eh = 1.55, gw = 2.3, cw = (W - 2 * MX - gw - 0.35 - 3 * 0.15) / 4;
      f.eval.forEach(([th, en, items], i) => { const ex = MX + i * (cw + 0.15);
        s.addShape(pres.shapes.RECTANGLE, { x: ex, y: ey, w: cw, h: eh, fill: { color: "F4F6F9" }, line: { color: "F4F6F9" }, objectName: `eval${i}` });
        T(s, th, { x: ex + 0.15, y: ey + 0.08, w: cw - 0.3, h: 0.36, fontSize: 15, bold: true });
        T(s, en, { x: ex + 0.15, y: ey + 0.43, w: cw - 0.3, h: 0.28, fontSize: 11, color: "2E5A88" });
        T(s, items.map((t, k) => ({ text: "· " + t, options: { breakLine: k < items.length - 1 } })), { x: ex + 0.15, y: ey + 0.75, w: cw - 0.3, h: 0.75, fontSize: 11, color: HEX.dk1, valign: "top" }); });
      const gx = W - MX - gw;
      s.addShape(pres.shapes.RIGHT_ARROW, { x: gx - 0.33, y: ey + eh / 2 - 0.13, w: 0.26, h: 0.26, fill: { color: NAVY }, line: { color: NAVY }, objectName: "arrow2" });
      s.addShape(pres.shapes.RECTANGLE, { x: gx, y: ey, w: gw, h: eh, fill: { color: NAVY }, line: { color: NAVY }, objectName: "goal" });
      T(s, f.goal[0], { x: gx + 0.15, y: ey + 0.15, w: gw - 0.3, h: 0.45, fontSize: 16, bold: true, color: HEX.lt1 });
      T(s, f.goal[1], { x: gx + 0.15, y: ey + 0.62, w: gw - 0.3, h: 0.8, fontSize: 12, color: "CADCFC", valign: "top" });
      break;
    }
    case "timeline_table": {
      const tot = d.segments.reduce((a, g) => a + g[2], 0), tw = W - 2 * MX, ty = 1.75, th = 0.95;
      const fills = ["F3D9A4", "BFD3EA", "2E5A88"];
      let tx = MX;
      d.segments.forEach(([lab, yrs, len, role], i) => { const w = (tw * len) / tot;
        s.addShape(pres.shapes.RECTANGLE, { x: tx, y: ty, w: w - 0.04, h: th, fill: { color: fills[i] }, line: { color: fills[i] }, objectName: `seg${i}` });
        const dark = i === 2;
        T(s, [{ text: `${lab} ${yrs}`, options: { bold: true, breakLine: true } }, { text: role, options: { fontSize: 11 } }],
          { x: tx + 0.12, y: ty + 0.08, w: w - 0.25, h: th - 0.15, fontSize: 14, color: dark ? HEX.lt1 : HEX.dk1, valign: "middle" });
        if (i > 0) { s.addShape(pres.shapes.LINE, { x: tx - 0.02, y: ty - 0.12, w: 0, h: th + 0.24, line: { color: "C0392B", width: 2, dashType: "dash" }, objectName: `purge${i}` }); }
        tx += w; });
      T(s, "เส้นประแดง = purge 5 วันที่ขอบ: ตัด target ที่คร่อมเส้นแบ่ง", { x: MX, y: ty + th + 0.15, w: tw, h: 0.3, fontSize: 11.5, color: "C0392B" });
      table(s, d.header, d.rows, MX, ty + th + 0.6, tw, 13);
      if (d.body) T(s, d.body, { x: MX, y: bottom - 0.6, w: tw, h: 0.6, fontSize: 14, color: HEX.dk2, bold: true, valign: "middle" });
      break;
    }
    case "figure_wide": {
      const im = fit(d.image, MX, top, W - 2 * MX, avail - 1.3);
      s.addImage({ ...im, y: top });
      bullets(s, d.items, MX, top + im.h + 0.2, W - 2 * MX, bottom - (top + im.h + 0.2), 17);
      break;
    }
    case "wide_plus": {
      const im = fit(d.image, MX, top, W - 2 * MX, 2.0);
      s.addImage({ ...im, y: top });
      const y2 = top + im.h + 0.15;
      const im2 = fit(d.image2, MX, y2, 4.6, bottom - y2);
      s.addImage({ ...im2, x: MX });
      bullets(s, d.items, MX + 5.0, y2 + 0.2, W - 2 * MX - 5.0, bottom - y2 - 0.2, 17);
      break;
    }
    case "two_figures": {
      const cw = (W - 2 * MX - 0.3) / 2;
      s.addImage(fit(d.image, MX, top, cw, avail));
      s.addImage(fit(d.image2, MX + cw + 0.3, top, cw, avail));
      break;
    }
    case "table": case "appendix_table": {
      const bodyH = d.body ? 0.75 : 0;
      table(s, d.header, d.rows, MX, top, W - 2 * MX, d.rows.length > 7 ? 13 : 15);
      if (d.body) T(s, d.body, { x: MX, y: bottom - bodyH, w: W - 2 * MX, h: bodyH, fontSize: 15, color: HEX.dk2, bold: true, valign: "middle" });
      break;
    }
    case "bullets":
      s.addShape(pres.shapes.RECTANGLE, { x: MX, y: top, w: 0.08, h: avail - 0.2, fill: { color: HEX.accent2 }, line: { color: HEX.accent2 }, objectName: "rule" });
      bullets(s, d.items, MX + 0.35, top, W - 2 * MX - 0.35, avail, 20);
      break;
  }
  source(s, d.source);
  if (d.notes) s.addNotes(d.notes);
}

(async () => {
  const out = "/home/claude/deliver/slides_updated_th.pptx";
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("written", out, SL.length);
})();
