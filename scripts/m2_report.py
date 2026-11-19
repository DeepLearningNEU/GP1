"""Build the Milestone 2 report (docs/milestone2_report.html -> .pdf) from the result files.

Observations are kept in docs/m2_observations.html (hand-written after looking at the results)
and inserted as-is. PDF is printed with headless Chrome, same as the proposal.
"""
import base64, csv, html, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "m2_caption"
DOCS = ROOT / "docs"
SAMPLE_IMG = RES / "sample_images"   # exported by m2_figures.py
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
LABEL = {"prefix": "Prefix conditioning", "cross": "Cross-attention"}


def img_tag(path, width):
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    ext = Path(path).suffix.lstrip(".").replace("jpg", "jpeg")
    return f'<img src="data:image/{ext};base64,{b64}" style="width:{width}">'


def thumb(name, size=150):
    """Embed the 300x300 copy exported by m2_figures.py (no image library needed)."""
    b64 = base64.b64encode((SAMPLE_IMG / Path(name).name).read_bytes()).decode()
    return f'<img src="data:image/jpeg;base64,{b64}" style="width:{size}px">'


def read_csv(p):
    return list(csv.DictReader(open(p))) if p.exists() else []


def table(rows, cols, fmt=None):
    fmt = fmt or {}
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{fmt.get(c, lambda v: v)(r[c])}</td>" for c in cols) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def main():
    cfg = {inj: json.load(open(RES / inj / "config.json")) for inj in ["prefix", "cross"] if (RES / inj / "config.json").exists()}
    ep = {inj: read_csv(RES / inj / "epoch_log.csv") for inj in cfg}
    dec = {inj: read_csv(RES / inj / "decoding_comparison.csv") for inj in cfg}
    samples = [json.loads(l) for l in open(RES / "prefix" / "samples_by_stage.jsonl")]
    stages = list(dict.fromkeys(s["stage"] for s in samples))
    by = {}
    for s in samples:
        by.setdefault(s["file_name"], {"reference": s["reference"]})[s["stage"]] = s["generated"]
    dcap = {}
    for r in map(json.loads, open(RES / "prefix" / "decoding_captions.jsonl")):
        if r["file_name"] in by:
            dcap.setdefault(r["file_name"], {})[r["strategy"]] = r["caption"]
    xcap = {}
    if (RES / "cross" / "samples_by_stage.jsonl").exists():
        for s in map(json.loads, open(RES / "cross" / "samples_by_stage.jsonl")):
            xcap.setdefault(s["file_name"], {})[s["stage"]] = s["generated"]
    c = cfg["prefix"]

    # ---- epoch log table
    ep_rows = []
    for inj in cfg:
        for r in ep[inj]:
            ep_rows.append({"Model": LABEL[inj], "Epoch": r["epoch"], "Step": r["step"],
                            "Train loss": r["train_loss"] or "—", "Val loss": r["val_loss"], "Val perplexity": r["val_ppl"],
                            "Time (min)": r["minutes"]})
    # ---- decoding tables
    dec_cols = ["strategy", "BLEU-1", "BLEU-4", "CIDEr", "CLIPScore", "avg_words", "distinct-2", "unique_captions",
                "repeated_bigram_rate", "sec_per_image"]
    pretty = {"strategy": "Strategy", "avg_words": "Avg words", "unique_captions": "Unique captions",
              "repeated_bigram_rate": "Repeated bigram", "sec_per_image": "Sec / image"}
    def dec_table(rows):
        rr = [{pretty.get(k, k): (v if k == "strategy" else f"{float(v):.3f}") for k, v in r.items() if k in dec_cols} for r in rows]
        return table(rr, [pretty.get(k, k) for k in dec_cols])

    # ---- sample table: image | untrained | step 100 | step 300 | final greedy | beam-5 | nucleus | reference
    final = stages[-1]
    sample_rows = ""
    for i, (name, s) in enumerate(by.items(), 1):
        d = dcap.get(name, {})
        lines = [("Untrained (step 0)", s.get("untrained", "")), ("Step 100", s.get("step 100", "")),
                 ("Step 300", s.get("step 300", "")), (f"Final ({final}), greedy", s.get(final, "")),
                 ("Final, beam-5", d.get("beam-5", "")), ("Final, nucleus p=0.9", d.get("nucleus p=0.9", ""))]
        if xcap.get(name):
            xs = list(xcap[name]); lines.append((f"Cross-attention ({xs[-1]}), greedy", xcap[name][xs[-1]]))
        lines.append(("Human reference", s["reference"]))
        cap_html = "".join(f"<div><span class='lab'>{html.escape(a)}:</span> {html.escape(b)}</div>" for a, b in lines)
        sample_rows += f"<tr><td class='num'>{i}</td><td class='im'>{thumb(name)}</td><td class='caps'>{cap_html}</td></tr>"

    obs = (DOCS / "m2_observations.html").read_text() if (DOCS / "m2_observations.html").exists() else "<p>(observations pending)</p>"
    cross_cfg = cfg.get("cross", {})
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>GP1 Milestone 2</title>
<style>
  @page {{ size: Letter; margin: 0.65in 0.7in; }}
  body {{ font-family: "Times New Roman", Times, serif; font-size: 10.5pt; line-height: 1.33; color: #000; margin: 0; }}
  .cover {{ height: 9.4in; display: flex; flex-direction: column; justify-content: center; text-align: center; page-break-after: always; }}
  .cover h1 {{ font-size: 21pt; margin: 0 0 10px; line-height: 1.25; }}
  .cover .kind {{ font-size: 15pt; margin-bottom: 44px; }}
  .cover .meta {{ font-size: 12.5pt; line-height: 1.8; }}
  h2 {{ font-size: 12pt; margin: 14px 0 5px; border-bottom: 1px solid #999; padding-bottom: 1px; }}
  h3 {{ font-size: 11pt; margin: 10px 0 3px; }}
  p {{ margin: 0 0 6px; text-align: justify; }}
  ul {{ margin: 0 0 6px; padding-left: 18px; }} li {{ margin: 0 0 3px; text-align: justify; }}
  table {{ width: 100%; border-collapse: collapse; margin: 4px 0 8px; font-size: 9.5pt; }}
  th, td {{ border: 1px solid #999; padding: 2px 5px; text-align: left; vertical-align: top; }}
  th {{ background: #eee; }}
  .samples td.im {{ width: 152px; }} .samples td.num {{ width: 14px; text-align: center; vertical-align: middle; }}
  .samples td.caps div {{ margin-bottom: 1px; }} .lab {{ color: #444; font-style: italic; }}
  .samples tr {{ page-break-inside: avoid; }}
  .fig {{ text-align: center; margin: 6px 0 2px; }} .cap {{ text-align: center; font-size: 9.5pt; color: #333; margin-bottom: 8px; }}
  .pb {{ page-break-before: always; }}
  code {{ font-family: Menlo, Consolas, monospace; font-size: 9pt; }}
</style></head><body>

<div class="cover">
  <h1>From Words to Worlds:<br>Image Captioning with a CLIP Encoder<br>and a GPT-2 Decoder</h1>
  <div class="kind">Milestone 2 — Model Integration and Baseline Captioning</div>
  <div class="meta">
    IE 7615 Deep Learning for AI<br>Generative Project<br><br>
    <b>Team members:</b> Zhiming Hu, Wan Luo<br>November 18, 2026<br><br>
    Code repository: github.com/DeepLearningNEU/GP1
  </div>
</div>

<h2>1. What was built</h2>
<ul>
  <li><b>Encoder:</b> CLIP ViT-L/14 vision Transformer, frozen. Features are pre-computed once: the projected [CLS] embedding (768-d) and the last-layer patch tokens (257 × 1024).</li>
  <li><b>Decoder:</b> GPT-2 small (124M), fine-tuned. Captions are fed as <code>&lt;|endoftext|&gt; caption &lt;|endoftext|&gt;</code>; the first token acts as BOS, the last as EOS, and the loss is computed on caption tokens and EOS only.</li>
  <li><b>Embedding injection — prefix conditioning:</b> a 2-layer MLP maps the CLIP embedding to {c['prefix_len']} prefix vectors (10 × 768) that are prepended to the caption token embeddings (ClipCap-style). GPT-2's architecture is unchanged; {c['params_new_M']}M new parameters.</li>
  <li><b>Embedding injection — cross-attention:</b> CLIP patch tokens are projected to 768-d (Linear + LayerNorm) and every GPT-2 block gets a new cross-attention layer over them ({cross_cfg.get('params_new_M', '—')}M new parameters).</li>
  <li><b>Data:</b> the 5,000 COCO train2017 images from Milestone 1, now with all of their captions ({c['train_pairs']:,} image–caption pairs); {c['val_images']} COCO val2017 images ({c['val_pairs']:,} captions) for validation loss, samples and the decoding study.</li>
  <li><b>Training:</b> AdamW, batch {c['bs']}, learning rate {c['lr_new']} for new layers and {c['lr_gpt']} for GPT-2, {c['warmup']} warm-up steps then linear decay, gradient clipping 1.0. Prefix model: {c['epochs']} epochs; cross-attention model: {cross_cfg.get('epochs', '—')} epochs. Trained on the NEU Explorer HPC cluster ({c.get('gpu', c['device'])}).</li>
  <li><b>Decoding:</b> greedy, beam search (3 and 5 beams, length-normalized), and nucleus sampling (p = 0.5 and 0.9), implemented directly so the same code works for both injection methods.</li>
</ul>

<h2>2. Training log</h2>
{table(ep_rows, ["Model", "Epoch", "Step", "Train loss", "Val loss", "Val perplexity", "Time (min)"])}
<p>Epoch 0 is the untrained model (randomly initialized mapping / cross-attention). Loss is per-token cross-entropy on the validation captions. Step-level logs are in <code>results/m2_caption/&lt;model&gt;/train_log.csv</code>; the full console log is <code>results/m2_caption/train_console.log</code>.</p>
<div class="fig">{img_tag(RES / "loss_curves.png", "72%")}</div>
<div class="cap">Figure 1. Training loss (every 50 steps) and validation loss (end of each epoch).</div>

<h2>3. Early caption samples</h2>
<p>Eight fixed validation images (not seen in training). Captions are recorded before training, early in training (steps 100 and 300), and at the end; the last three lines compare decoding strategies and the cross-attention model.</p>
<table class="samples">{sample_rows}</table>

<h2 class="pb">4. Decoding strategies (500 validation images)</h2>
<h3>Prefix conditioning</h3>
{dec_table(dec["prefix"])}
{"<h3>Cross-attention</h3>" + dec_table(dec["cross"]) if dec.get("cross") else ""}
<p>BLEU and CIDEr are computed against the five human captions per image (pycocoevalcap, lower-cased, punctuation removed). CLIPScore = 2.5 × max(cos(image, caption), 0) with CLIP ViT-L/14, a reference-free image–text match. Distinct-2 = unique bigrams / all bigrams across the 500 captions; unique captions = share of distinct captions; repeated bigram = share of captions that repeat a bigram.</p>

<h2>5. Summary of observations</h2>
{obs}
</body></html>"""
    out = DOCS / "milestone2_report.html"
    out.write_text(doc)
    pdf = DOCS / "milestone2_report.pdf"
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", out.as_uri()],
                   check=True, capture_output=True)
    print("wrote", out, "and", pdf)


if __name__ == "__main__":
    main()
