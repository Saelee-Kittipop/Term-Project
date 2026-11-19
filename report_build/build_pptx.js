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
  const secTitle = d.kind === "title" || d.kind === "agenda" ? "เปิด" : d.kind === "closing" ? "สรุป" : (d.section || "").split(" ").slice(0, 2).join(" ") || "เปิด";
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
