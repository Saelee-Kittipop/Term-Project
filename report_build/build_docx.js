// Render report_spec.json into a Thai DOCX (Sarabun) with figures and tables.
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, LevelFormat, BorderStyle, PageBreak, Footer, PageNumber, TableLayoutType,
} = require("docx");

const spec = JSON.parse(fs.readFileSync("/home/claude/deliver/report_spec.json", "utf8"));
const FONT = "Sarabun";
const F = { ascii: FONT, hAnsi: FONT, cs: FONT, eastAsia: FONT };
const PAGE_W = 11906, MARGIN = 1300, CONTENT = PAGE_W - 2 * MARGIN;   // A4

const run = (text, o = {}) => new TextRun({
  text, font: F, size: o.size || 22, sizeComplexScript: o.size || 22, bold: !!o.bold, boldComplexScript: !!o.bold,
  italics: !!o.italic, italicsComplexScript: !!o.italic, color: o.color, language: { value: "en-US", bidirectional: "th-TH" },
});

function pngSize(path) {
  const b = fs.readFileSync(path);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}

const children = [];
const P = (t, o = {}) => new Paragraph({ children: [run(t, o)], alignment: o.align || AlignmentType.JUSTIFIED, spacing: { after: o.after ?? 120, line: 300 }, keepNext: o.keepNext });

function table(b) {
  const n = b.header.length;
  let w = b.widths && b.widths.length === n ? b.widths.slice() : Array(n).fill(Math.floor(CONTENT / n));
  const s = w.reduce((a, c) => a + c, 0);
  w = w.map((x) => Math.floor((x * CONTENT) / s));
  w[n - 1] += CONTENT - w.reduce((a, c) => a + c, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (t, i, head) => new TableCell({
    width: { size: w[i], type: WidthType.DXA }, borders,
    shading: head ? { fill: "E8EEF4", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children: [new Paragraph({ children: [run(String(t), { size: 18, bold: head })], alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })],
  });
  children.push(P(b.caption, { bold: true, size: 20, keepNext: true, align: AlignmentType.LEFT, after: 60 }));
  children.push(new Table({
    width: { size: CONTENT, type: WidthType.DXA }, columnWidths: w, layout: TableLayoutType.FIXED,
    rows: [new TableRow({ tableHeader: true, children: b.header.map((h, i) => cell(h, i, true)) }),
      ...b.rows.map((r) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false)) }))],
  }));
  if (b.note) children.push(P(b.note, { size: 17, color: "595959", align: AlignmentType.LEFT, after: 20 }));
  children.push(P(b.source, { size: 16, color: "7F7F7F", align: AlignmentType.LEFT, after: 200 }));
}

function figure(b) {
  const img = pngSize(b.path);
  const wIn = Math.min(b.width || 6.3, CONTENT / 1440);
  const wPx = wIn * 96, hPx = (wPx * img.h) / img.w;
  children.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: img.data, transformation: { width: wPx, height: hPx },
      altText: { title: b.caption, description: b.caption, name: b.fid } })] }));
  children.push(P(b.caption, { bold: true, size: 19, align: AlignmentType.CENTER, after: 30 }));
  children.push(P(b.source, { size: 16, color: "7F7F7F", align: AlignmentType.CENTER, after: 200 }));
}

for (const b of spec) {
  switch (b.type) {
    case "cover":
      children.push(new Paragraph({ spacing: { before: 2400 }, children: [] }));
      b.lines.forEach((t, i) => { const big = i === 1 || i === 2;
        children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: i === 2 ? 240 : 140 },
        children: [run(t, { size: big ? 36 : i === 0 ? 28 : i === 3 ? 25 : 22, bold: big || i === 0, color: big ? "1F3864" : undefined })] })); });
      children.push(new Paragraph({ children: [new PageBreak()] }));
      break;
    case "h1": children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [run(b.text, { size: 32, bold: true, color: "1F3864" })] })); break;
    case "h2": children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [run(b.text, { size: 26, bold: true, color: "2E5A88" })] })); break;
    case "h3": children.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [run(b.text, { size: 23, bold: true })] })); break;
    case "p": children.push(P(b.text)); break;
    case "eq": children.push(P(b.text, { italic: true, align: AlignmentType.CENTER, size: 21 })); break;
    case "bullets": b.items.forEach((t) => children.push(new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 290 }, children: [run(t)] }))); break;
    case "table": table(b); break;
    case "figure": figure(b); break;
  }
}

const doc = new Document({
  creator: "Term Project (drafted with Claude)", title: "รายงานการทดลอง: การพยากรณ์ความผันผวนพอร์ต ETF ด้วย LSTM และ Transformer",
  styles: {
    default: { document: { run: { font: F, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: F, size: 32, bold: true }, paragraph: { spacing: { before: 120, after: 200 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: F, size: 26, bold: true }, paragraph: { spacing: { before: 240, after: 120 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: F, size: 23, bold: true }, paragraph: { spacing: { before: 160, after: 80 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: 1300, bottom: 1300, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: F, size: 18 })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync("/home/claude/deliver/report_updated_th.docx", buf); console.log("written", buf.length); });
