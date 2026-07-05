# Export VoxCPM2 component graphs to NCNN.
#
# Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
# Licensed under MIT. Not all rights reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pnnx
import torch
import torch.nn.functional as F
from torch import nn
from voxcpm import VoxCPM
from voxcpm.modules.minicpm4.model import apply_rotary_pos_emb

HEAD_DIM = 128


COMPONENTS = [
    "text_embed",
    "base_decoder_step",
    "residual_decoder_step",
    "fsq",
    "fusion_proj",
    "dit_proj",
    "stop_head",
    "feat_encoder",
    "dit_estimator",
    "audio_vae_encoder",
    "audio_vae_decoder",
]

COMPONENT_CHOICES = COMPONENTS


def pnnx_inputshape(inputs: tuple[torch.Tensor, ...]) -> str:
    parts: list[str] = []
    for tensor in inputs:
        if tensor.dtype in (torch.long, torch.int64):
            dtype = "i64"
        elif tensor.dtype in (torch.int, torch.int32):
            dtype = "i32"
        else:
            dtype = "f32"
        shape = ",".join(str(dim) for dim in tensor.shape)
        parts.append(f"[{shape}]{dtype}")
    return ",".join(parts)


def sanitize_param(param_path: Path) -> None:
    text = param_path.read_text(encoding="utf-8", errors="replace")
    text = text.replace("F.scaled_dot_product_attention", "SDPA")
    param_path.write_text(text, encoding="utf-8")


def fix_dit_prefix_param(param_path: Path) -> None:
    lines = param_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    skip_blob = None
    for line in lines:
        fields = line.split()
        if fields and fields[0] == "MemoryData" and "0=512" in fields and "1=1" in fields:
            fields = [f for f in fields if f != "1=1"]
            line = "%-24s %-24s %s" % (fields[0], fields[1], " ".join(fields[2:]))
        elif fields and fields[0] == "Concat" and fields[1] in {"cat_0", "cat_1"}:
            fields[-1] = "0=0"
            line = "%-24s %-24s %s" % (fields[0], fields[1], " ".join(fields[2:]))
        elif fields and fields[0] == "Reshape" and "in1" in fields and "0=1024" in fields and "1=2" in fields:
            fields = [f for f in fields if f != "2=1"]
            line = "%-24s %-24s %s" % (fields[0], fields[1], " ".join(fields[2:]))
        elif fields and fields[0] == "ExpandDims" and len(fields) >= 7 and fields[4] == "34":
            fields[-1] = "-23303=1,0"
            line = "%-24s %-24s %s" % (fields[0], fields[1], " ".join(fields[2:]))
        elif fields and fields[0] == "Concat" and fields[1] == "cat_2":
            if fields[3] != "1":
                raise RuntimeError(f"unexpected dit_prefix cat_2 line: {line}")
            skip_blob = fields[-2]
            fields[-2] = "out0"
            fields[-1] = "0=0"
            line = "%-24s %-24s %s" % (fields[0], fields[1], " ".join(fields[2:]))
        elif fields and fields[0] == "Squeeze" and skip_blob is not None and skip_blob in fields:
            continue
        out.append(line)

    if skip_blob is not None and len(out) >= 2:
        layer_count, blob_count = map(int, out[1].split()[:2])
        out[1] = f"{layer_count - 1} {blob_count - 1}"
    param_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def fix_dit_time_reshape_param(param_path: Path, expected: int = 3) -> None:
    lines = param_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    fixed = 0
    for line in lines:
        fields = line.split()
        if fields and fields[0] == "ExpandDims" and len(fields) >= 7:
            if fields[4] in {"in2", "in4"} and fields[-1] == "-23303=1,1":
                fields[0] = "Reshape"
                fields[-1] = "0=1"
                line = "%-24s %-24s %s" % (fields[0], fields[1], " ".join(fields[2:]))
                fixed += 1
            elif fields[-1] == "-23303=1,0":
                fields[0] = "Reshape"
                fields[-1:] = ["0=1024", "1=1"]
                line = "%-24s %-24s %s" % (fields[0], fields[1], " ".join(fields[2:]))
                fixed += 1
        out.append(line)

    if fixed != expected:
        raise RuntimeError(f"expected to fix {expected} DiT time reshape ops in {param_path}, fixed {fixed}")
    param_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def add_sdpa_kvcache(param_path: Path) -> None:
    raw = param_path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    if len(lines) < 2 or lines[0].strip() != "7767517":
        raise RuntimeError(f"not an ncnn param file: {param_path}")

    body = lines[2:]
    input_insert_idx = -1
    for i, line in enumerate(body):
        fields = line.split()
        if fields and fields[0] == "Input":
            input_insert_idx = i
    if input_insert_idx < 0:
        raise RuntimeError(f"no Input layer found in {param_path}")

    scale = f"{1.0 / (HEAD_DIM ** 0.5):.7g}"
    sdpa_i = 0
    out_body: list[str] = []
    for line in body:
        fields = line.split()
        if fields and fields[0] == "SDPA":
            name = fields[1]
            nin = int(fields[2])
            nout = int(fields[3])
            blobs = fields[4 : 4 + nin + nout]
            inputs = blobs[:nin]
            outputs = blobs[nin:]
            params = fields[4 + nin + nout :]
            if nout == 3:
                out_body.append(line)
                sdpa_i += 1
                continue
            if nin != 4 or nout != 1:
                raise RuntimeError(f"unexpected SDPA signature in {param_path}: {line}")
            pd: dict[str, str] = {}
            for p in params:
                key, value = p.split("=", 1)
                pd[key] = value
            pd["5"] = "1"
            pd["6"] = scale
            pd["7"] = "1"
            new_inputs = inputs + [f"cache_k{sdpa_i}", f"cache_v{sdpa_i}"]
            new_outputs = outputs + [f"out_cache_k{sdpa_i}", f"out_cache_v{sdpa_i}"]
            pstr = " ".join(f"{key}={pd[key]}" for key in sorted(pd, key=lambda x: int(x)))
            out_body.append(
                "%-24s %-24s %d %d %s %s"
                % ("SDPA", name, len(new_inputs), len(new_outputs), " ".join(new_inputs + new_outputs), pstr)
            )
            sdpa_i += 1
        else:
            out_body.append(line)

    if sdpa_i == 0:
        raise RuntimeError(f"no SDPA layer found in {param_path}")

    cache_blobs = [name for i in range(sdpa_i) for name in (f"cache_k{i}", f"cache_v{i}")]
    input_line = "%-24s %-24s 0 %d %s" % ("Input", "kv_cache", len(cache_blobs), " ".join(cache_blobs))
    out_body.insert(input_insert_idx + 1, input_line)

    layer_lines = [line for line in out_body if line.strip()]
    layer_count = len(layer_lines)
    blob_count = sum(int(line.split()[3]) for line in layer_lines)
    param_path.write_text("\n".join([lines[0], f"{layer_count} {blob_count}", *out_body]) + "\n", encoding="utf-8")


def run_pnnx(pt_path: Path, inputs: tuple[torch.Tensor, ...], out_dir: Path, fp16: bool) -> None:
    prefix = out_dir / pt_path.stem
    result = subprocess.run(
        [
            pnnx.EXEC_PATH,
            str(pt_path),
            f"inputshape={pnnx_inputshape(inputs)}",
            "device=cpu",
            "optlevel=2",
            f"fp16={1 if fp16 else 0}",
            f"pnnxparam={prefix.with_suffix('.pnnx.param')}",
            f"pnnxbin={prefix.with_suffix('.pnnx.bin')}",
            f"pnnxpy={prefix.parent / f'{prefix.name}_pnnx.py'}",
            "pnnxonnx=/dev/null",
            f"ncnnparam={prefix.with_suffix('.ncnn.param')}",
            f"ncnnbin={prefix.with_suffix('.ncnn.bin')}",
            f"ncnnpy={prefix.parent / f'{prefix.name}_ncnn.py'}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout, end="")
        print(result.stderr, end="")
        result.check_returncode()
    sanitize_param(prefix.with_suffix(".ncnn.param"))


def export_module(name: str, module: nn.Module, inputs: tuple[torch.Tensor, ...], out_dir: Path, fp16: bool) -> None:
    module = module.eval().to(torch.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / f"{name}.pt"
    with torch.inference_mode():
        module(*inputs)
        traced = torch.jit.trace(module, inputs, check_trace=False)
        traced.save(str(pt_path))
    run_pnnx(pt_path, inputs, out_dir, fp16)
    if name == "dit_prefix":
        fix_dit_prefix_param(out_dir / f"{name}.ncnn.param")
        fix_dit_time_reshape_param(out_dir / f"{name}.ncnn.param")
    if name == "dit_estimator":
        fix_dit_time_reshape_param(out_dir / f"{name}.ncnn.param")
    if name.endswith("_decoder_step"):
        add_sdpa_kvcache(out_dir / f"{name}.ncnn.param")


class TextEmbed(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embed = model.base_lm.embed_tokens
        self.scale = 1.0 if not getattr(model.config.lm_config, "use_mup", False) else model.config.lm_config.scale_emb

    def forward(self, ids):
        return (self.embed(ids) * self.scale).squeeze(0)


class Decoder(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, inputs_embeds):
        hidden, _ = self.lm(inputs_embeds=inputs_embeds, is_causal=True)
        return hidden


def decoder_forward_with_mask(lm, inputs_embeds, attn_mask):
    if lm.rope_emb is not None:
        position_ids = torch.arange(0, inputs_embeds.size(1), dtype=torch.long, device=inputs_embeds.device)
        position_emb = lm.rope_emb(position_ids)
    else:
        position_emb = None

    hidden_states = inputs_embeds
    for decoder_layer in lm.layers:
        residual = hidden_states
        normed = decoder_layer.input_layernorm(hidden_states)
        attn = decoder_layer.self_attn
        bsz, q_len, _ = normed.size()

        query_states = attn.q_proj(normed)
        key_states = attn.k_proj(normed)
        value_states = attn.v_proj(normed)

        query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

        if position_emb is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, *position_emb)

        attn_output = F.scaled_dot_product_attention(
            query_states.contiguous(),
            key_states.contiguous(),
            value_states.contiguous(),
            attn_mask=attn_mask,
            enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, attn.num_heads * attn.head_dim)
        attn_output = attn.o_proj(attn_output)

        if decoder_layer.use_mup:
            hidden_states = residual + attn_output * (
                decoder_layer.scale_depth / (decoder_layer.num_hidden_layers**0.5)
            )
        else:
            hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        if decoder_layer.use_mup:
            hidden_states = residual + hidden_states * (
                decoder_layer.scale_depth / (decoder_layer.num_hidden_layers**0.5)
            )
        else:
            hidden_states = residual + hidden_states

    return lm.norm(hidden_states)


class MaskedDecoder(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, inputs_embeds, attn_mask):
        return decoder_forward_with_mask(self.lm, inputs_embeds, attn_mask)


class DecoderStep(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, inputs_embeds, attn_mask, cos=None, sin=None):
        hidden_states = inputs_embeds
        position_emb = None if self.lm.rope_emb is None else (cos, sin)

        for decoder_layer in self.lm.layers:
            residual = hidden_states
            normed = decoder_layer.input_layernorm(hidden_states)
            attn = decoder_layer.self_attn
            bsz, _ = normed.size()

            query_states = attn.q_proj(normed)
            key_states = attn.k_proj(normed)
            value_states = attn.v_proj(normed)

            query_states = query_states.view(bsz, 1, attn.num_heads, attn.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, 1, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, 1, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

            if position_emb is not None:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, *position_emb)

            attn_output = F.scaled_dot_product_attention(
                query_states.contiguous(),
                key_states.contiguous(),
                value_states.contiguous(),
                attn_mask=attn_mask,
                enable_gqa=True,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, attn.num_heads * attn.head_dim)
            attn_output = attn.o_proj(attn_output)

            if decoder_layer.use_mup:
                hidden_states = residual + attn_output * (
                    decoder_layer.scale_depth / (decoder_layer.num_hidden_layers**0.5)
                )
            else:
                hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer.mlp(hidden_states)
            if decoder_layer.use_mup:
                hidden_states = residual + hidden_states * (
                    decoder_layer.scale_depth / (decoder_layer.num_hidden_layers**0.5)
                )
            else:
                hidden_states = residual + hidden_states

        return self.lm.norm(hidden_states)


class Fsq(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fsq = model.fsq_layer

    def forward(self, hidden):
        return self.fsq(hidden)


class FusionProj(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.proj = model.fusion_concat_proj

    def forward(self, lm_hidden, feat_embed):
        return self.proj(torch.cat((lm_hidden, feat_embed), dim=-1))


class DitProj(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.lm_proj = model.lm_to_dit_proj
        self.res_proj = model.res_to_dit_proj

    def forward(self, lm_hidden, residual_hidden):
        return torch.cat((self.lm_proj(lm_hidden), self.res_proj(residual_hidden)), dim=-1)


class StopHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.proj = model.stop_proj
        self.act = model.stop_actn
        self.head = model.stop_head

    def forward(self, hidden):
        return self.head(self.act(self.proj(hidden)))


class FeatEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.feat_encoder = model.feat_encoder
        self.enc_to_lm_proj = model.enc_to_lm_proj

    def forward(self, feat):
        return self.enc_to_lm_proj(self.feat_encoder(feat))


class DitEstimator(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.estimator = model.feat_decoder.estimator

    def forward(self, x, mu, t, cond, dt):
        out = self.estimator(x, mu, t, cond, dt)
        return out.reshape(out.size(0), -1)


class DitPrefix(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.estimator = model.feat_decoder.estimator

    def forward(self, x, mu, t, cond, dt):
        x = self.estimator.in_proj(x.transpose(1, 2).contiguous())
        cond = self.estimator.cond_proj(cond.transpose(1, 2).contiguous())
        t = self.estimator.time_mlp(self.estimator.time_embeddings(t).to(x.dtype))
        dt = self.estimator.delta_time_mlp(self.estimator.time_embeddings(dt).to(x.dtype))
        t = t + dt
        mu = mu.view(x.size(0), -1, x.size(-1))
        return torch.cat([mu, t.unsqueeze(1), cond, x], dim=1).squeeze(0)


class DitCore(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.estimator = model.feat_decoder.estimator

    def forward(self, hidden):
        hidden = hidden.unsqueeze(0)
        out, _ = self.estimator.decoder(inputs_embeds=hidden, is_causal=False)
        out = out[:, 7:, :]
        out = self.estimator.out_proj(out)
        return out.transpose(1, 2).contiguous().reshape(out.size(0), -1)


class DitLayer(nn.Module):
    def __init__(self, model, layer_index: int):
        super().__init__()
        self.decoder = model.feat_decoder.estimator.decoder
        layer = self.decoder.layers[layer_index]
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.q_proj = layer.self_attn.q_proj
        self.k_proj = layer.self_attn.k_proj
        self.v_proj = layer.self_attn.v_proj
        self.o_proj = layer.self_attn.o_proj
        self.mlp = layer.mlp
        self.num_heads = layer.self_attn.num_heads
        self.num_key_value_heads = layer.self_attn.num_key_value_heads
        self.head_dim = layer.self_attn.head_dim
        self.use_mup = layer.use_mup
        self.scale_depth = layer.scale_depth
        self.num_hidden_layers = layer.num_hidden_layers

    def forward(self, hidden):
        position_ids = torch.arange(0, hidden.size(0), dtype=torch.long, device=hidden.device)
        position_emb = self.decoder.rope_emb(position_ids)
        residual = hidden
        normed = self.input_layernorm(hidden)
        q = self.q_proj(normed)
        k = self.k_proj(normed)
        v = self.v_proj(normed)
        q = q.view(1, hidden.size(0), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(1, hidden.size(0), self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(1, hidden.size(0), self.num_key_value_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_emb)
        attn_out = F.scaled_dot_product_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            attn_mask=None,
            enable_gqa=True,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(hidden.size(0), self.num_heads * self.head_dim)
        attn_out = self.o_proj(attn_out)
        if self.use_mup:
            hidden = residual + attn_out * (self.scale_depth / (self.num_hidden_layers**0.5))
        else:
            hidden = residual + attn_out

        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        if self.use_mup:
            hidden = residual + hidden * (self.scale_depth / (self.num_hidden_layers**0.5))
        else:
            hidden = residual + hidden
        return hidden


class DitOut(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.estimator = model.feat_decoder.estimator

    def forward(self, hidden):
        hidden = hidden.unsqueeze(0)
        out = self.estimator.decoder.norm(hidden)
        out = out[:, 7:, :]
        out = self.estimator.out_proj(out)
        return out.transpose(1, 2).contiguous().reshape(out.size(0), -1)


class AudioVaeEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.audio_vae = model.audio_vae
        self.sample_rate = model._encode_sample_rate

    def forward(self, wav):
        return self.audio_vae.encode(wav, self.sample_rate)


class AudioVaeDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.audio_vae = model.audio_vae

    def forward(self, latent):
        return self.audio_vae.decode(latent)


def make_decoder_mask(decoder_len: int) -> torch.Tensor:
    mask = torch.full((decoder_len, decoder_len), -10000.0, dtype=torch.float32)
    for i in range(decoder_len):
        mask[i, : i + 1] = 0.0
    return mask


def component_module(model, name: str, decoder_len: int, masked_decoder: bool) -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    if name == "text_embed":
        return TextEmbed(model), (torch.randint(0, 512, (1, 4), dtype=torch.long),)
    if name == "base_decoder":
        if masked_decoder:
            return (
                MaskedDecoder(model.base_lm),
                (torch.randn(1, decoder_len, model.config.lm_config.hidden_size), make_decoder_mask(decoder_len)),
            )
        return Decoder(model.base_lm), (torch.randn(1, decoder_len, model.config.lm_config.hidden_size),)
    if name == "base_decoder_step":
        return (
            DecoderStep(model.base_lm),
            (
                torch.randn(1, model.config.lm_config.hidden_size),
                torch.zeros(1, 1, dtype=torch.float32),
                torch.randn(1, HEAD_DIM),
                torch.randn(1, HEAD_DIM),
            ),
        )
    if name == "residual_decoder":
        if masked_decoder:
            return (
                MaskedDecoder(model.residual_lm),
                (torch.randn(1, decoder_len, model.config.lm_config.hidden_size), make_decoder_mask(decoder_len)),
            )
        return Decoder(model.residual_lm), (torch.randn(1, decoder_len, model.config.lm_config.hidden_size),)
    if name == "residual_decoder_step":
        return (
            DecoderStep(model.residual_lm),
            (
                torch.randn(1, model.config.lm_config.hidden_size),
                torch.zeros(1, 1, dtype=torch.float32),
            ),
        )
    if name == "fsq":
        return Fsq(model), (torch.randn(1, 2048),)
    if name == "fusion_proj":
        return FusionProj(model), (torch.randn(1, 2048), torch.randn(1, 2048))
    if name == "dit_proj":
        return DitProj(model), (torch.randn(1, 2048), torch.randn(1, 2048))
    if name == "stop_head":
        return StopHead(model), (torch.randn(1, 2048),)
    if name == "feat_encoder":
        return FeatEncoder(model), (torch.randn(1, 1, model.patch_size, model.feat_dim),)
    if name == "dit_estimator":
        return (
            DitEstimator(model),
            (
                torch.randn(1, model.feat_dim, model.patch_size),
                torch.randn(1, 2048),
                torch.rand(1),
                torch.randn(1, model.feat_dim, model.patch_size),
                torch.zeros(1),
            ),
        )
    if name == "dit_prefix":
        return (
            DitPrefix(model),
            (
                torch.randn(1, model.feat_dim, model.patch_size),
                torch.randn(1, 2048),
                torch.rand(1),
                torch.randn(1, model.feat_dim, model.patch_size),
                torch.zeros(1),
            ),
        )
    if name == "dit_core":
        return DitCore(model), (torch.randn(11, 1024),)
    if name.startswith("dit_layer_"):
        layer_index = int(name.rsplit("_", 1)[1])
        return DitLayer(model, layer_index), (torch.randn(11, 1024),)
    if name == "dit_out":
        return DitOut(model), (torch.randn(11, 1024),)
    if name == "audio_vae_encoder":
        length = model.patch_size * model.chunk_size * 2
        return AudioVaeEncoder(model), (torch.randn(1, length),)
    if name == "audio_vae_decoder":
        length = model.patch_size * 2
        return AudioVaeDecoder(model), (torch.randn(1, model.audio_vae.latent_dim, length),)
    raise ValueError(f"unknown component: {name}")


def update_manifest(asset_dir: Path, exported: list[str], decoder_len: int) -> None:
    model_json = asset_dir / "model.json"
    manifest = json.loads(model_json.read_text(encoding="utf-8"))
    params = manifest.setdefault("params", {})
    for name in exported:
        params[name] = {
            "param": f"{name}.ncnn.param",
            "bin": f"{name}.ncnn.bin",
        }
    missing = [name for name in manifest.get("missing_components", []) if name not in exported]
    manifest["missing_components"] = missing
    model_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export VoxCPM2 component ncnn graphs for voxcpm2-ncnn.")
    parser.add_argument("--model-id", default="openbmb/VoxCPM2")
    parser.add_argument("--asset-dir", type=Path, default=Path("assets/voxcpm2"))
    parser.add_argument("--components", nargs="+", default=COMPONENTS, choices=COMPONENT_CHOICES)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    pipeline = VoxCPM.from_pretrained(
        args.model_id,
        load_denoiser=False,
        optimize=False,
        device="cpu",
        local_files_only=True,
    )
    model = pipeline.tts_model.eval()
    model.config.dtype = "float32"
    model = model.to(torch.float32)

    exported: list[str] = []
    for name in args.components:
        print(f"exporting {name}", flush=True)
        module, inputs = component_module(model, name, 4, False)
        export_module(name, module, inputs, args.asset_dir, args.fp16)
        exported.append(name)

    update_manifest(args.asset_dir, exported, 4)
    print(f"updated {args.asset_dir / 'model.json'}")


if __name__ == "__main__":
    main()
