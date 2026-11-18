"""Decoding strategies for ClipCaptioner: greedy, beam search, nucleus (top-p) sampling.

All functions take `cond` = model.condition(...) for a batch of images and return token-id lists
(without BOS/EOS). Generation stops at <|endoftext|> or max_len tokens.
"""
import torch


def _cond_rows(cond, idx):
    return cond[idx]


@torch.no_grad()
def greedy(model, cond, max_len=30):
    B, dev = cond.shape[0], cond.device
    ids = torch.full((B, 1), model.bos, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for _ in range(max_len):
        nxt = model.next_token_logits(cond, ids).argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, model.eos), nxt)
        ids = torch.cat([ids, nxt[:, None]], 1)
        done |= nxt == model.eos
        if done.all():
            break
    return [_strip(r, model.eos) for r in ids[:, 1:].tolist()]


@torch.no_grad()
def nucleus(model, cond, top_p=0.9, temperature=1.0, max_len=30, generator=None):
    B, dev = cond.shape[0], cond.device
    ids = torch.full((B, 1), model.bos, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for _ in range(max_len):
        logits = model.next_token_logits(cond, ids) / temperature
        probs = logits.softmax(-1)
        sp, si = probs.sort(-1, descending=True)
        keep = sp.cumsum(-1) - sp < top_p            # smallest set whose mass reaches top_p (always keeps the top token)
        sp = sp * keep
        choice = torch.multinomial((sp / sp.sum(-1, keepdim=True)).cpu(), 1, generator=generator).to(dev)
        nxt = si.gather(-1, choice).squeeze(-1)
        nxt = torch.where(done, torch.full_like(nxt, model.eos), nxt)
        ids = torch.cat([ids, nxt[:, None]], 1)
        done |= nxt == model.eos
        if done.all():
            break
    return [_strip(r, model.eos) for r in ids[:, 1:].tolist()]


@torch.no_grad()
def beam_search(model, cond, beam=5, max_len=30, length_penalty=1.0):
    """Beam search, batched over images (B*beam rows per forward pass).

    Each image keeps `beam` live hypotheses; a hypothesis that emits EOS is moved to that image's
    finished list with score sum(log p) / len**length_penalty. An image stops once it has `beam`
    finished hypotheses whose best score cannot be beaten (or max_len is reached).
    """
    B, K, dev = cond.shape[0], beam, cond.device
    cond_k = cond.repeat_interleave(K, dim=0)                          # (B*K, ...)
    seqs = torch.full((B * K, 1), model.bos, dtype=torch.long, device=dev)
    scores = torch.full((B, K), float("-inf"), device=dev)
    scores[:, 0] = 0.0                                                 # only one live beam at t=0
    finished = [[] for _ in range(B)]
    active = torch.ones(B, dtype=torch.bool, device=dev)
    for t in range(max_len):
        logp = model.next_token_logits(cond_k, seqs).log_softmax(-1).view(B, K, -1)
        V = logp.shape[-1]
        total = (scores[..., None] + logp).view(B, -1)                 # (B, K*V)
        top_s, top_i = total.topk(2 * K, dim=-1)                        # 2K so we can drop EOS candidates
        src, tok = top_i // V, top_i % V
        new_seqs = seqs.new_full((B, K, seqs.shape[1] + 1), model.eos)
        new_scores = torch.full((B, K), float("-inf"), device=dev)
        prev = seqs.view(B, K, -1)
        for b in range(B):
            if not active[b]:
                continue
            j = 0
            for s, k, w in zip(top_s[b].tolist(), src[b].tolist(), tok[b].tolist()):
                if s == float("-inf"):
                    break
                if w == model.eos:
                    finished[b].append((prev[b, k, 1:].tolist(), s / (t + 1) ** length_penalty))
                    continue
                if j < K:
                    new_seqs[b, j, :-1], new_seqs[b, j, -1], new_scores[b, j] = prev[b, k], w, s
                    j += 1
            if len(finished[b]) >= K:
                best_done = max(sc for _, sc in finished[b])
                best_live = new_scores[b].max().item() / (t + 2) ** length_penalty
                if best_live <= best_done:
                    active[b] = False
        seqs, scores = new_seqs.view(B * K, -1), new_scores
        if not active.any():
            break
    out = []
    prev = seqs.view(B, K, -1)
    for b in range(B):
        if not finished[b]:   # nothing ended with EOS within max_len: fall back to the best live beam
            finished[b] = [(prev[b, k, 1:].tolist(), scores[b, k].item() / max_len ** length_penalty) for k in range(K)]
        out.append(_strip(max(finished[b], key=lambda x: x[1])[0], model.eos))
    return out


def _strip(ids, eos):
    return ids[:ids.index(eos)] if eos in ids else ids


def decode(model, cond, strategy="greedy", **kw):
    if strategy == "greedy":
        return greedy(model, cond, **kw)
    if strategy == "beam":
        return beam_search(model, cond, **kw)
    if strategy == "nucleus":
        return nucleus(model, cond, **kw)
    raise ValueError(strategy)
