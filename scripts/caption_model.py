"""CLIP image encoder + GPT-2 language decoder for image captioning.

Two ways to inject the image into GPT-2:

  prefix  (ClipCap-style prefix conditioning)
          CLIP [CLS] image embedding (768) --MLP--> k prefix vectors (k x 768) that are
          prepended to the caption token embeddings. GPT-2 attends to them with its normal
          causal self-attention; no change to the GPT-2 architecture.

  cross   (cross-attention)
          CLIP patch tokens (257 x 1024) --Linear+LayerNorm--> 257 x 768 "encoder states".
          GPT-2 is built with add_cross_attention=True, so every block gets a new (randomly
          initialised) cross-attention layer that attends from caption tokens to the patches.

CLIP is frozen in both cases; its features are pre-computed (scripts/extract_clip_features.py).

Fine-tuning strategy for the pre-trained GPT-2 weights (`finetune`):
  full    all GPT-2 weights are trained (Milestone 2)
  lora    GPT-2 frozen; LoRA adapters (rank r) on the self-attention projections c_attn / c_proj
  frozen  GPT-2 frozen; only the new image layers train (mapper, and cross-attention in cross mode).
          With prefix injection this is prefix tuning: the image-conditioned prefix is the only thing learned.
Sequence layout fed to GPT-2:  [prefix (prefix mode only)] <|endoftext|> caption tokens <|endoftext|>
The first <|endoftext|> acts as BOS, the last one as EOS; loss is computed on caption tokens + EOS.
"""
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

GPT2_NAME = "openai-community/gpt2"


class ClipCaptioner(nn.Module):
    def __init__(self, injection="prefix", prefix_len=10, clip_dim=768, patch_dim=1024,
                 gpt2_name=GPT2_NAME, freeze_gpt2=False, finetune="full", lora_r=8):
        super().__init__()
        assert injection in ("prefix", "cross")
        self.injection, self.prefix_len = injection, prefix_len
        if injection == "prefix":
            self.gpt = GPT2LMHeadModel.from_pretrained(gpt2_name)
            d = self.gpt.config.n_embd
            hidden = d * prefix_len // 2
            self.mapper = nn.Sequential(nn.Linear(clip_dim, hidden), nn.Tanh(), nn.Linear(hidden, d * prefix_len))
        else:
            cfg = GPT2Config.from_pretrained(gpt2_name, add_cross_attention=True)
            self.gpt = GPT2LMHeadModel.from_pretrained(gpt2_name, config=cfg)
            d = self.gpt.config.n_embd
            self.mapper = nn.Sequential(nn.Linear(patch_dim, d), nn.LayerNorm(d))
        self.bos = self.eos = self.gpt.config.eos_token_id
        assert finetune in ("full", "lora", "frozen")
        self.finetune = "frozen" if freeze_gpt2 else finetune
        if self.finetune != "full":  # freeze pre-trained GPT-2; new cross-attention layers stay trainable
            for n, p in self.gpt.named_parameters():
                p.requires_grad = "crossattention" in n or "ln_cross_attn" in n
        if self.finetune == "lora":
            from peft import LoraConfig, inject_adapter_in_model
            cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, fan_in_fan_out=True,
                             target_modules=r".*\.h\.\d+\.attn\.(c_attn|c_proj)")  # self-attention only
            self.gpt = inject_adapter_in_model(cfg, self.gpt)  # adds trainable lora_A / lora_B
            for n, p in self.gpt.named_parameters():   # injection freezes non-adapter weights; re-open new layers
                if "crossattention" in n or "ln_cross_attn" in n:
                    p.requires_grad = True

    # ------------------------------------------------------------------ conditioning
    def condition(self, clip_emb=None, patches=None):
        """Image features -> what GPT-2 consumes: prefix embeddings (B,k,d) or encoder states (B,257,d)."""
        if self.injection == "prefix":
            return self.mapper(clip_emb).view(clip_emb.shape[0], self.prefix_len, -1)
        return self.mapper(patches)

    def _run(self, cond, input_ids, attention_mask=None, labels=None):
        tok = self.gpt.transformer.wte(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if self.injection == "prefix":
            B, k = cond.shape[:2]
            embeds = torch.cat([cond, tok], dim=1)
            attention_mask = torch.cat([attention_mask.new_ones(B, k), attention_mask], dim=1)
            if labels is not None:
                labels = torch.cat([labels.new_full((B, k), -100), labels], dim=1)
            return self.gpt(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels)
        return self.gpt(inputs_embeds=tok, attention_mask=attention_mask,
                        encoder_hidden_states=cond, labels=labels)

    def forward(self, cond, input_ids, attention_mask, labels):
        """Teacher-forced loss. `cond` comes from self.condition()."""
        return self._run(cond, input_ids, attention_mask, labels)

    @torch.no_grad()
    def next_token_logits(self, cond, ids):
        """Logits for the next token given the image condition and the tokens so far (B, T)."""
        return self._run(cond, ids).logits[:, -1, :]
