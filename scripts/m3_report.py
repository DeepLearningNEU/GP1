"""Build the Milestone 3 report (docs/milestone3_report.html -> .pdf) from results/m3/ (tables + figures made by
notebooks/03_evaluation.ipynb). Section text that interprets the numbers lives in docs/m3_text.json
({section_key: html}), written after looking at the results. PDF is printed with headless Chrome.
"""
import base64, csv, html, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES, DOCS = ROOT / "results" / "m3", ROOT / "docs"
FIG, TAB, QI = RES / "figures", RES / "tables", RES / "qual_images"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def img(path, width):
    return f'<img src="data:image/png;base64,{base64.b64encode(Path(path).read_bytes()).decode()}" style="width:{width}">'


def thumb(name, size=110):
    b64 = base64.b64encode((QI / Path(name).name).read_bytes()).decode()
    return f'<img src="data:image/jpeg;base64,{b64}" style="width:{size}px">'


def rows(name):
    return list(csv.DictReader(open(TAB / name)))


def fmt(v):
    """3 decimals for non-integer numbers, everything else as text."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return html.escape(str(v))
    dec = str(v).split(".")[1] if "." in str(v) else ""
    return f"{f:.3f}" if len(dec) > 3 else str(v)


def table(rs, cols, bold_max=()):
    best = {c: max(float(r[c]) for r in rs) for c in bold_max}
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = ""
    for r in rs:
        cells = ""
        for c in cols:
            v = fmt(r[c])
            if c in best and float(r[c]) == best[c]:
                v = f"<b>{v}</b>"
            cells += f"<td>{v}</td>"
        body += f"<tr>{cells}</tr>"
    return f"<table><tr>{head}</tr>{body}</table>"


def fig(name, width, caption):
    return f'<div class="fig">{img(FIG / name, width)}</div><div class="cap">{caption}</div>'


def main():
    T = json.load(open(DOCS / "m3_text.json"))
    cfg = {p.parent.name: json.load(open(p)) for p in RES.glob("*/config.json")}
    c0 = cfg["prefix_full"]
    order = lambda r: (r["Injection"] != "Cross-attention", ["Full fine-tuning", "LoRA", "Prefix tuning (frozen GPT-2)"].index(r["Fine-tuning"]))
    train = sorted(rows("training_summary.csv"), key=order)
    main_t = sorted(rows("finetuning_comparison_beam3.csv"), key=order)
    gpus = {inj: cfg[f"{inj}_full"].get("gpu", "") for inj in ("prefix", "cross")}
    dec = rows("decoding_strategies.csv")
    qual = rows("qualitative_examples.csv")
    qs = json.load(open(TAB / "qualitative_summary.json"))
    M = ["BLEU-1", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "CLIPScore"]

    def dec_rows(inj):
        out = []
        for r in dec:
            if r["injection"] == inj:
                out.append({"Decoding": r["strategy"].replace("nucleus:0.5:1.0", "nucleus p=0.5").replace("nucleus:0.9:1.0", "nucleus p=0.9")
                            .replace("sample:", "sampling T=").replace("beam:", "beam "),
                            **{k: r[k] for k in M}, "distinct-2": r["distinct-2"], "Unique": r["unique_captions"], "Sec/img": r["sec_per_image"]})
        return out

    rand_rows = ""
    for q in [q for q in qual if q["group"] == "random"]:
        rand_rows += (f"<tr><td class='im'>{thumb(q['file_name'], 92)}</td><td>"
                      f"<span class='lab'>greedy:</span> {html.escape(q['greedy'])}<br>"
                      f"<span class='lab'>beam-3:</span> {html.escape(q['beam:3'])}<br>"
                      f"<span class='lab'>nucleus p=0.9:</span> {html.escape(q['nucleus:0.9:1.0'])}<br>"
                      f"<span class='lab'>reference:</span> {html.escape(q['reference'])}</td>"
                      f"<td>{float(q['CLIPScore']):.3f}</td><td>{float(q['CIDEr']):.3f}</td></tr>")
    qual_rows = ""
    for q in [q for q in qual if q["group"] != "random"]:
        qual_rows += (f"<tr><td class='im'>{thumb(q['file_name'], 92)}</td><td><i>{html.escape(q['group'])}</i><br>"
                      f"<span class='lab'>beam-3:</span> {html.escape(q['beam:3'])}<br>"
                      f"<span class='lab'>reference:</span> {html.escape(q['reference'])}</td>"
                      f"<td>{float(q['CLIPScore']):.3f}</td><td>{float(q['CIDEr']):.3f}</td></tr>")

    ft_table = table([{"Strategy": a_, "What is trained": b_} for a_, b_ in [
        ("Full fine-tuning", "all GPT-2 weights (learning rate 5e-5) and the new image layers (2e-4)"),
        ("LoRA", f"GPT-2 frozen; rank-{c0['lora_r']} LoRA adapters (alpha = {2 * c0['lora_r']}, dropout 0.05) on every "
                 f"self-attention c_attn / c_proj, learning rate {c0['lr_lora']}; plus the new image layers"),
        ("Prefix tuning (frozen GPT-2)", "GPT-2 completely frozen; only the new image layers. With prefix injection, "
                                         "the image-conditioned prefix is the only thing learned (prefix tuning)")]],
        ["Strategy", "What is trained"])
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>GP1 Milestone 3</title>
<style>
  @page {{ size: Letter; margin: 0.65in 0.7in; }}
  body {{ font-family: "Times New Roman", Times, serif; font-size: 10.5pt; line-height: 1.33; color: #000; margin: 0; }}
  .cover {{ height: 9.4in; display: flex; flex-direction: column; justify-content: center; text-align: center; page-break-after: always; }}
  .cover h1 {{ font-size: 21pt; margin: 0 0 10px; line-height: 1.25; }}
  .cover .kind {{ font-size: 15pt; margin-bottom: 44px; }}
  .cover .meta {{ font-size: 12.5pt; line-height: 1.8; }}
  h2 {{ font-size: 13pt; margin: 14px 0 5px; border-bottom: 1px solid #999; padding-bottom: 1px; }}
  h3 {{ font-size: 11pt; margin: 10px 0 3px; }}
  p {{ margin: 0 0 6px; text-align: justify; }}
  ul {{ margin: 0 0 6px; padding-left: 18px; }} li {{ margin: 0 0 3px; text-align: justify; }}
  table {{ width: 100%; border-collapse: collapse; margin: 4px 0 6px; font-size: 9pt; page-break-inside: avoid; }}
  th, td {{ border: 1px solid #999; padding: 2px 4px; text-align: left; vertical-align: top; }}
  th {{ background: #eee; }}
  td.im {{ width: 94px; }} .lab {{ color: #444; font-style: italic; }}
  .fig {{ text-align: center; margin: 6px 0 2px; page-break-inside: avoid; }}
  .cap {{ text-align: center; font-size: 9.5pt; color: #333; margin-bottom: 8px; }}
  .pb {{ page-break-before: always; }}
  code {{ font-family: Menlo, Consolas, monospace; font-size: 8.8pt; }}
</style></head><body>

<div class="cover">
  <h1>From Words to Worlds:<br>Image Captioning with a CLIP Encoder<br>and a GPT-2 Decoder</h1>
  <div class="kind">Milestone 3 — Evaluation and Analysis</div>
  <div class="meta">
    IE 7615 Deep Learning for AI<br>Generative Project<br><br>
    <b>Team members:</b> Zhiming Hu, Wan Luo<br>{T['date']}<br><br>
    Code repository: github.com/DeepLearningNEU/GP1<br>Evaluation notebook: <code>notebooks/03_evaluation.ipynb</code>
  </div>
</div>

<h2>1. Methods</h2>
<h3>1.1 Model</h3>
<p>The captioner encodes each image with the CLIP ViT-L/14 vision Transformer, which stays frozen; its features are pre-computed once. GPT-2 small (124M) generates the caption. Captions are fed to GPT-2 as
<code>&lt;|endoftext|&gt; caption &lt;|endoftext|&gt;</code>, and the loss is the cross-entropy on the caption tokens and the end token. The image reaches the decoder through one of two <b>embedding-injection methods</b>:</p>
<ul>
  <li><b>Prefix conditioning</b> (ClipCap-style): a 2-layer MLP maps the 768-d CLIP image embedding to {c0['prefix_len']} vectors in GPT-2's input space, which are prepended to the caption tokens. GPT-2 attends to them through its ordinary causal self-attention.</li>
  <li><b>Cross-attention</b>: the 257 CLIP patch tokens (1024-d) are projected to 768-d. Every GPT-2 block gets a new cross-attention layer that attends from the caption tokens to these patches.</li>
</ul>
<h3>1.2 Fine-tuning strategies</h3>
<p>For each injection method we compare three ways of training the pre-trained GPT-2 weights. The new image layers (the MLP mapper, or the projection plus the cross-attention layers) are always trained.</p>
{ft_table}
<h3>1.3 Data and training protocol</h3>
<ul>
  <li><b>Training set:</b> 5,000 COCO train2017 images with all of their captions ({c0['train_pairs']:,} image–caption pairs).</li>
  <li><b>Model-selection set:</b> COCO val2017 images 1,001–5,000. After every epoch we compute the validation loss on these images and keep the checkpoint with the lowest loss. This early stopping fixes the overfitting we observed in Milestone 2.</li>
  <li><b>Test set:</b> the first 1,000 val2017 images, each with 5 human reference captions. These images are not used for training or model selection, and every number in Section 2 is computed on them.</li>
  <li><b>Optimization:</b> AdamW, batch {c0['bs']}, {c0['warmup']} warm-up steps then linear decay, gradient clipping 1.0, at most {c0['epochs']} epochs. All 6 models were trained on NEU Explorer HPC GPUs: the prefix models on a {html.escape(gpus['prefix'])}, the cross-attention models on a {html.escape(gpus['cross'])}.</li>
</ul>
<h3>1.4 Decoding strategies</h3>
<ul>
  <li><b>Greedy:</b> take the most likely token at each step.</li>
  <li><b>Beam search:</b> keep the K best hypotheses, K &isin; {{2, 3, 5, 8, 10}}, with length-normalized scores.</li>
  <li><b>Temperature sampling:</b> sample from softmax(logits / T) with T &isin; {{0.25, 0.5, 0.75, 1.0, 1.25, 1.5}}.</li>
  <li><b>Nucleus (top-p) sampling:</b> p &isin; {{0.5, 0.9}} at T = 1.</li>
</ul>
<p>All decoding settings were run with the two fully fine-tuned models; the LoRA and prefix-tuning models were decoded with greedy, beam-3 and nucleus p = 0.9. Sampling uses a fixed random seed, and generation stops at the end token or after 30 tokens.</p>
<h3>1.5 Metrics</h3>
<ul>
  <li><b>BLEU-1..4, METEOR, ROUGE-L, CIDEr</b> (pycocoevalcap). Each generated caption is compared with its 5 references after lower-casing and removing punctuation.
    <ul>
      <li>BLEU measures n-gram precision.</li>
      <li>METEOR matches words, including stems and synonyms, and balances recall against precision.</li>
      <li>ROUGE-L uses the longest common subsequence.</li>
      <li>CIDEr is a consensus metric: TF-IDF-weighted n-gram similarity, so words that are informative across the dataset count more.</li>
    </ul></li>
  <li><b>Image–caption alignment:</b> CLIPScore = 2.5 &middot; max(cos(CLIP image, CLIP caption), 0), computed with CLIP ViT-L/14. It needs no references, so it measures whether the caption describes the image. We also inspect captions by hand, including the highest- and lowest-scoring ones.</li>
  <li><b>Diversity:</b> distinct-2, the number of unique bigrams divided by all bigrams across the 1,000 captions; and the share of captions that are unique.</li>
</ul>

<h2 class="pb">2. Results</h2>
<h3>2.1 Training</h3>
{table(train, ["Injection", "Fine-tuning", "Trainable params (M)", "Best epoch", "Best val loss", "Val loss @ epoch 5", "Train min"])}
<div class="cap">Table 1. Training summary. Val loss is the per-token loss on the 4,000 model-selection images.</div>
{fig("val_loss_by_strategy.png", "88%", "Figure 1. Validation loss on the model-selection images after each epoch.")}
{T['training']}

<h3>2.2 Fine-tuning strategies and injection methods</h3>
{table(main_t, ["Injection", "Fine-tuning", "Trainable params (M)"] + M, bold_max=M)}
<div class="cap">Table 2. Test-set results with beam search (width 3). Bold marks the best value in each column.</div>
{fig("finetuning_comparison.png", "96%", "Figure 2. CIDEr, METEOR and CLIPScore by fine-tuning strategy and injection method (beam-3, 1,000 test images).")}
{T['finetuning']}

<h3>2.3 Parameter sensitivity</h3>
<p><b>Beam width.</b></p>
{fig("sensitivity_beam_width.png", "100%", "Figure 3. Effect of beam width (1 = greedy) for the two fully fine-tuned models.")}
{T['beam']}
<p><b>Temperature.</b></p>
{fig("sensitivity_temperature.png", "100%", "Figure 4. Effect of the sampling temperature (pure sampling, top-p = 1).")}
{T['temperature']}
<p><b>Injection method.</b> {T['injection']}</p>

<h3>2.4 Decoding strategies compared</h3>
{fig("accuracy_vs_diversity.png", "74%", "Figure 5. Accuracy (CIDEr) against diversity (distinct-2) for every decoding setting. Labels are shown for the cross-attention model.")}
<p><b>Prefix conditioning, full fine-tuning</b></p>
{table(dec_rows("prefix"), ["Decoding"] + M + ["distinct-2", "Unique", "Sec/img"], bold_max=M)}
<div class="cap">Table 3. Every decoding setting, prefix conditioning (full fine-tuning). Bold marks the best value in each column.</div>
<p><b>Cross-attention, full fine-tuning</b></p>
{table(dec_rows("cross"), ["Decoding"] + M + ["distinct-2", "Unique", "Sec/img"], bold_max=M)}
<div class="cap">Table 4. Every decoding setting, cross-attention (full fine-tuning).</div>
{T['decoding']}

<h3>2.5 Qualitative alignment between image and caption</h3>
<p>The examples below use the best model by CIDEr ({html.escape(qs['best_tag'].replace('_', ', '))}, beam-3). Across the 1,000 test images, the per-image correlation between the reference-free CLIPScore and the reference-based CIDEr is r = {qs['corr_cider_clipscore']:.2f}.</p>
<table><tr><th>Image</th><th>Generated captions and one human reference</th><th>CLIPScore</th><th>CIDEr</th></tr>{rand_rows}</table>
<div class="cap">Table 5. Eight random test images (scores are for the beam-3 caption). The same examples as an image grid are in <code>results/m3/figures/qualitative_grid.png</code>.</div>
<table><tr><th>Image</th><th>Caption (beam-3) and reference</th><th>CLIPScore</th><th>CIDEr</th></tr>{qual_rows}</table>
<div class="cap">Table 6. Test images with the highest and lowest CLIPScore.</div>
{fig("cider_vs_clipscore.png", "50%", "Figure 6. Per-image CIDEr against CLIPScore (beam-3).")}
{T['qualitative']}

<h3>2.6 Summary of findings</h3>
{T['summary']}
</body></html>"""
    out = DOCS / "milestone3_report.html"
    out.write_text(doc)
    pdf = DOCS / "milestone3_report.pdf"
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", out.as_uri()],
                   check=True, capture_output=True)
    print("wrote", out, "and", pdf)


if __name__ == "__main__":
    main()
