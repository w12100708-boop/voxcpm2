# Export VoxCPM2 component graphs to NCNN.
#
# Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
# Licensed under MIT. Not all rights reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
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
    "base_decoder_kv",
    "residual_decoder_kv",
    "fsq",
    "fusion_proj",
    "dit_proj",
    "stop_head",
    "feat_encoder",
    "dit_estimator",
    "audio_vae_encoder",
    "audio_vae_decoder",
]

DUAL_BACKEND_COMPONENTS = frozenset({"dit_estimator", "base_decoder_kv", "residual_decoder_kv", "feat_encoder"})


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


def rename_dit_sdpa_param(param_path: Path) -> None:
    lines = param_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    fixed = 0
    existing = 0
    for line in lines:
        fields = line.split()
        if fields and fields[0] == "SDPA":
            fields[0] = "VoxCPM2SDPA"
            line = "%-24s %-24s %s" % (fields[0], fields[1], " ".join(fields[2:]))
            fixed += 1
        elif fields and fields[0] == "VoxCPM2SDPA":
            existing += 1
        out.append(line)
    if fixed == 0 and existing == 0:
        raise RuntimeError(f"no SDPA layer found in {param_path}")
    param_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def fuse_dit_timestep_embedding_param(param_path: Path) -> None:
    lines = param_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2 or lines[0].strip() != "7767517":
        raise RuntimeError(f"not an ncnn param file: {param_path}")

    body = lines[2:]
    out: list[str] = []
    i = 0
    fused = 0

    def is_layer(line: str, layer_type: str, bottom: str | None = None) -> bool:
        fields = line.split()
        if not fields or fields[0] != layer_type:
            return False
        if bottom is not None:
            nin = int(fields[2])
            return bottom in fields[4 : 4 + nin]
        return True

    while i < len(body):
        fields = body[i].split()
        if fields and fields[0] == "Reshape" and len(fields) >= 7 and fields[4] in {"in2", "in4"}:
            source = fields[4]
            consumed = 9 if source == "in2" else 7
            branch = body[i : i + consumed]
            if len(branch) != consumed:
                raise RuntimeError(f"incomplete timestep branch in {param_path}: {body[i]}")

            if source == "in2":
                checks = [
                    is_layer(branch[0], "Reshape", "in2"),
                    is_layer(branch[1], "MemoryData"),
                    is_layer(branch[2], "Split"),
                    is_layer(branch[3], "BinaryOp"),
                    is_layer(branch[4], "BinaryOp"),
                    is_layer(branch[5], "Split"),
                    is_layer(branch[6], "UnaryOp"),
                    is_layer(branch[7], "UnaryOp"),
                    is_layer(branch[8], "Concat"),
                ]
                concat_fields = branch[8].split()
            else:
                checks = [
                    is_layer(branch[0], "Reshape", "in4"),
                    is_layer(branch[1], "BinaryOp"),
                    is_layer(branch[2], "BinaryOp"),
                    is_layer(branch[3], "Split"),
                    is_layer(branch[4], "UnaryOp"),
                    is_layer(branch[5], "UnaryOp"),
                    is_layer(branch[6], "Concat"),
                ]
                concat_fields = branch[6].split()

            if not all(checks):
                raise RuntimeError(f"unexpected timestep branch in {param_path}: {body[i]}")
            if concat_fields[0] != "Concat" or int(concat_fields[2]) != 2 or int(concat_fields[3]) != 1:
                raise RuntimeError(f"unexpected timestep concat in {param_path}: {' '.join(concat_fields)}")

            output_blob = concat_fields[4 + int(concat_fields[2])]
            if source == "in2":
                # Keep the folded frequency table layer so the unchanged .bin
                # stream stays aligned for all following Gemm weights.
                out.append(branch[1])
            out.append(
                "%-24s %-24s %s"
                % (
                    "VoxCPM2TimestepEmbedding",
                    f"timestep_embedding_{fused}",
                    f"1 1 {source} {output_blob} 0=512 1=1000.0 2=10000.0 3={1 if source == 'in2' else 0}",
                )
            )
            fused += 1
            i += consumed
            continue

        out.append(body[i])
        i += 1

    if fused != 2:
        raise RuntimeError(f"expected to fuse 2 DiT timestep embedding branches in {param_path}, fused {fused}")

    layer_lines = [line for line in out if line.strip()]
    layer_count = len(layer_lines)
    blob_count = sum(int(line.split()[3]) for line in layer_lines)
    param_path.write_text("\n".join([lines[0], f"{layer_count} {blob_count}", *out]) + "\n", encoding="utf-8")


def force_dit_gemm_output_fp32_param(param_path: Path) -> None:
    lines = param_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    fixed = 0
    for line in lines:
        fields = line.split()
        if fields and fields[0] == "Gemm":
            for i, field in enumerate(fields):
                if field.startswith("13="):
                    fields[i] = "13=1"
                    break
            else:
                fields.append("13=1")
            line = "%-24s %-24s %s" % (fields[0], fields[1], " ".join(fields[2:]))
            fixed += 1
        out.append(line)
    if fixed == 0:
        raise RuntimeError(f"no Gemm layer found in {param_path}")
    param_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def format_param_line(fields: list[str]) -> str:
    return "%-24s %-24s %s" % (fields[0], fields[1], " ".join(fields[2:]))


def write_ncnn_param(path: Path, magic: str, body: list[str]) -> None:
    layer_lines = [line for line in body if line.strip()]
    layer_count = len(layer_lines)
    blob_count = sum(int(line.split()[3]) for line in layer_lines)
    path.write_text("\n".join([magic, f"{layer_count} {blob_count}", *layer_lines]) + "\n", encoding="utf-8")


def add_rmsnorm_fp32_flags(body: list[str]) -> list[str]:
    out: list[str] = []
    fixed = 0
    for line in body:
        fields = line.split()
        if fields and fields[0] == "RMSNorm":
            for i, field in enumerate(fields):
                if field.startswith("31="):
                    fields[i] = "31=3"
                    break
            else:
                fields.append("31=3")
            line = format_param_line(fields)
            fixed += 1
        out.append(line)

    if fixed == 0:
        raise RuntimeError("no RMSNorm layer found in component param")
    return out


def normalize_cpu_param_body(body: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    restored = 0
    while i < len(body):
        fields = body[i].split()
        if fields and fields[0] == "VoxCPM2DTypeAdapter" and i + 2 < len(body):
            rms_fields = body[i + 1].split()
            cast_fields = body[i + 2].split()
            if (
                len(fields) >= 6
                and rms_fields
                and rms_fields[0] == "RMSNorm"
                and cast_fields
                and cast_fields[0] == "Cast"
                and int(fields[2]) == 1
                and int(fields[3]) == 1
                and int(rms_fields[2]) == 1
                and int(rms_fields[3]) == 1
                and int(cast_fields[2]) == 1
                and int(cast_fields[3]) == 1
                and rms_fields[4] == fields[5]
                and cast_fields[4] == rms_fields[5]
            ):
                rms_fields[4] = fields[4]
                rms_fields[5] = cast_fields[5]
                out.append(format_param_line(rms_fields))
                restored += 1
                i += 3
                continue
        out.append(body[i])
        i += 1

    if restored:
        print(f"restored {restored} RMSNorm adapter blocks to CPU fp32 graph")
    return out


def make_vulkan_param_body(cpu_body: list[str]) -> list[str]:
    out: list[str] = []
    fixed = 0
    for line in add_rmsnorm_fp32_flags(cpu_body):
        fields = line.split()
        if fields and fields[0] == "RMSNorm":
            bottom_count = int(fields[2])
            top_count = int(fields[3])
            if bottom_count != 1 or top_count != 1:
                raise RuntimeError(f"unexpected RMSNorm arity: {line}")

            bottom = fields[4]
            top = fields[5]
            adapter_top = f"{bottom}_dtype_adapter_{fixed}"
            rms_top = f"{top}_dtype_rms_{fixed}"

            out.append(format_param_line([
                "VoxCPM2DTypeAdapter",
                f"{fields[1]}_dtype_adapter",
                "1",
                "1",
                bottom,
                adapter_top,
            ]))

            rms_fields = fields.copy()
            rms_fields[4] = adapter_top
            rms_fields[5] = rms_top
            out.append(format_param_line(rms_fields))

            out.append(format_param_line([
                "Cast",
                f"{fields[1]}_cast_out_fp16",
                "1",
                "1",
                rms_top,
                top,
                "0=1",
                "1=2",
            ]))
            fixed += 1
            continue

        out.append(line)

    if fixed == 0:
        raise RuntimeError("no RMSNorm layer found in component param")
    return out


def write_backend_params(param_path: Path, cpu_name: str, vulkan_name: str) -> None:
    """Split a single ncnn param into CPU and Vulkan variants.

    The Vulkan variant wraps every RMSNorm in a fp16->fp32->fp16 adapter
    triple so the surrounding fp16 graph keeps numerically safe RMSNorm.
    """
    lines = param_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2 or lines[0].strip() != "7767517":
        raise RuntimeError(f"not an ncnn param file: {param_path}")

    normalized_body = normalize_cpu_param_body(lines[2:])
    cpu_body = add_rmsnorm_fp32_flags(normalized_body)
    vulkan_body = make_vulkan_param_body(normalized_body)
    write_ncnn_param(param_path.with_name(cpu_name), lines[0], cpu_body)
    write_ncnn_param(param_path.with_name(vulkan_name), lines[0], vulkan_body)


def write_component_backend_params(param_path: Path, name: str) -> None:
    """Generate <name>.cpu.ncnn.param and <name>.vulkan.ncnn.param."""
    write_backend_params(
        param_path,
        f"{name}.cpu.ncnn.param",
        f"{name}.vulkan.ncnn.param",
    )
    param_path.unlink()


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


def run_pnnx(
    pt_path: Path,
    inputshape: str,
    out_dir: Path,
    fp16: bool,
    inputshape2: str | None = None,
) -> None:
    prefix = out_dir / pt_path.stem
    cmd = [
        pnnx.EXEC_PATH,
        str(pt_path),
        f"inputshape={inputshape}",
    ]
    if inputshape2 is not None:
        cmd.append(f"inputshape2={inputshape2}")
    cmd.extend(
        [
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
        ]
    )
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout, end="")
        print(result.stderr, end="")
        result.check_returncode()
    sanitize_param(prefix.with_suffix(".ncnn.param"))


def trace_module(
    name: str,
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    out_dir: Path,
) -> Path:
    module = module.eval().to(torch.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / f"{name}.pt"
    with torch.inference_mode():
        module(*inputs)
        traced = torch.jit.trace(module, inputs, check_trace=False)
        traced.save(str(pt_path))
    return pt_path


def postprocess_exported_param(name: str, out_dir: Path) -> None:
    param_path = out_dir / f"{name}.ncnn.param"
    if name == "dit_estimator":
        fuse_dit_timestep_embedding_param(param_path)
        force_dit_gemm_output_fp32_param(param_path)
        rename_dit_sdpa_param(param_path)
    elif name in {"base_decoder_kv", "residual_decoder_kv"}:
        add_sdpa_kvcache(param_path)

    if name in DUAL_BACKEND_COMPONENTS:
        write_component_backend_params(param_path, name)


def cleanup_intermediates(name: str, out_dir: Path) -> None:
    for path in [
        out_dir / f"{name}.pt",
        out_dir / f"{name}.pnnx.param",
        out_dir / f"{name}.pnnx.bin",
        out_dir / f"{name}_pnnx.py",
        out_dir / f"{name}_ncnn.py",
        out_dir / f"{name}.foldable_constants.zip",
    ]:
        path.unlink(missing_ok=True)


class TextEmbed(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embed = model.base_lm.embed_tokens
        self.scale = 1.0 if not getattr(model.config.lm_config, "use_mup", False) else model.config.lm_config.scale_emb

    def forward(self, ids):
        return (self.embed(ids) * self.scale).squeeze(0)


class DecoderKv(nn.Module):
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
        half_dim = self.estimator.config.hidden_size // 2
        scale = -math.log(10000) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, dtype=torch.float32) * scale)
        self.register_buffer("time_freq", freq, persistent=False)
        self.register_buffer("dit_position_ids", torch.arange(0, 11, dtype=torch.long), persistent=False)

    def time_embedding(self, sample: torch.Tensor) -> torch.Tensor:
        sample = sample.reshape(sample.size(0), 1)
        emb = 1000.0 * sample * self.time_freq.reshape(1, self.time_freq.numel())
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

    def linear3(self, linear: nn.Linear, hidden: torch.Tensor) -> torch.Tensor:
        return linear(hidden.reshape(-1, hidden.size(-1))).reshape(hidden.size(0), hidden.size(1), -1)

    def mlp(self, mlp: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        gate = self.linear3(mlp.gate_proj, hidden)
        up = self.linear3(mlp.up_proj, hidden)
        return self.linear3(mlp.down_proj, mlp.act_fn(gate) * up)

    def decoder(self, hidden: torch.Tensor) -> torch.Tensor:
        position_emb = self.estimator.decoder.rope_emb(self.dit_position_ids)
        for layer in self.estimator.decoder.layers:
            residual = hidden
            normed = layer.input_layernorm(hidden)
            q = self.linear3(layer.self_attn.q_proj, normed)
            k = self.linear3(layer.self_attn.k_proj, normed)
            v = self.linear3(layer.self_attn.v_proj, normed)
            q = q.reshape(hidden.size(0), hidden.size(1), layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(1, 2)
            k = k.reshape(hidden.size(0), hidden.size(1), layer.self_attn.num_key_value_heads, layer.self_attn.head_dim).transpose(1, 2)
            v = v.reshape(hidden.size(0), hidden.size(1), layer.self_attn.num_key_value_heads, layer.self_attn.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, *position_emb)
            attn = F.scaled_dot_product_attention(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                attn_mask=None,
                is_causal=False,
                enable_gqa=True,
            )
            attn = attn.transpose(1, 2).contiguous().reshape(hidden.size(0), hidden.size(1), layer.self_attn.num_heads * layer.self_attn.head_dim)
            attn = self.linear3(layer.self_attn.o_proj, attn)
            if layer.use_mup:
                hidden = residual + attn * (layer.scale_depth / (layer.num_hidden_layers**0.5))
            else:
                hidden = residual + attn

            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = self.mlp(layer.mlp, hidden)
            if layer.use_mup:
                hidden = residual + hidden * (layer.scale_depth / (layer.num_hidden_layers**0.5))
            else:
                hidden = residual + hidden
        return self.estimator.decoder.norm(hidden)

    def forward(self, x, mu, t, cond, dt):
        batch = x.size(0)
        x = x.transpose(1, 2).contiguous()
        x = self.estimator.in_proj(x.reshape(-1, x.size(-1))).reshape(batch, -1, self.estimator.config.hidden_size)
        cond = cond.transpose(1, 2).contiguous()
        cond = self.estimator.cond_proj(cond.reshape(-1, cond.size(-1))).reshape(batch, -1, self.estimator.config.hidden_size)
        prefix = cond.size(1)

        t = self.estimator.time_mlp(self.time_embedding(t).to(x.dtype))
        dt = self.estimator.delta_time_mlp(self.time_embedding(dt).to(x.dtype))
        time_token = (t + dt).reshape(x.size(0), 1, x.size(-1))

        hidden_size = x.size(-1)
        mu = mu.reshape(-1, hidden_size)
        time_token = time_token.reshape(-1, hidden_size)
        cond = cond.reshape(-1, hidden_size)
        x = x.reshape(-1, hidden_size)
        hidden0 = torch.cat([mu[0:2, :], time_token[0:1, :]], dim=0)
        hidden0 = torch.cat([hidden0, cond[0:4, :]], dim=0)
        hidden0 = torch.cat([hidden0, x[0:4, :]], dim=0).reshape(1, 11, hidden_size)
        hidden1 = torch.cat([mu[2:4, :], time_token[1:2, :]], dim=0)
        hidden1 = torch.cat([hidden1, cond[4:8, :]], dim=0)
        hidden1 = torch.cat([hidden1, x[4:8, :]], dim=0).reshape(1, 11, hidden_size)
        hidden0 = self.decoder(hidden0).reshape(11, hidden_size)[7:11, :]
        hidden1 = self.decoder(hidden1).reshape(11, hidden_size)[7:11, :]
        hidden = torch.cat([hidden0, hidden1], dim=0)
        out = self.estimator.out_proj(hidden).reshape(batch, 4, -1)
        out = out.transpose(1, 2).contiguous()
        return out.reshape(out.size(0), -1)


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


def component_module(model, name: str, decoder_len: int) -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    if name == "text_embed":
        return TextEmbed(model), (torch.randint(0, 512, (1, 4), dtype=torch.long),)
    if name == "base_decoder_kv":
        return (
            DecoderKv(model.base_lm),
            (
                torch.randn(1, decoder_len, model.config.lm_config.hidden_size),
                make_decoder_mask(decoder_len),
                torch.randn(decoder_len, HEAD_DIM),
                torch.randn(decoder_len, HEAD_DIM),
            ),
        )
    if name == "residual_decoder_kv":
        return (
            DecoderKv(model.residual_lm),
            (
                torch.randn(1, decoder_len, model.config.lm_config.hidden_size),
                make_decoder_mask(decoder_len),
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
                torch.randn(2, model.feat_dim, model.patch_size),
                torch.randn(2, 2048),
                torch.rand(2),
                torch.randn(2, model.feat_dim, model.patch_size),
                torch.zeros(2),
            ),
        )
    if name == "audio_vae_encoder":
        length = model.patch_size * model.chunk_size * 2
        return AudioVaeEncoder(model), (torch.randn(1, length),)
    if name == "audio_vae_decoder":
        length = model.patch_size * 2
        return AudioVaeDecoder(model), (torch.randn(1, model.audio_vae.latent_dim, length),)
    raise ValueError(f"unknown component: {name}")


def component_shape2_inputs(model, name: str) -> tuple[torch.Tensor, ...] | None:
    hidden_size = model.config.lm_config.hidden_size
    if name == "base_decoder_kv":
        return (
            torch.randn(1, 1, hidden_size),
            make_decoder_mask(1),
            torch.randn(1, HEAD_DIM),
            torch.randn(1, HEAD_DIM),
        )
    if name == "residual_decoder_kv":
        return (
            torch.randn(1, 1, hidden_size),
            make_decoder_mask(1),
        )
    return None


def update_manifest(asset_dir: Path, exported: list[str], decoder_len: int) -> None:
    model_json = asset_dir / "model.json"
    manifest = json.loads(model_json.read_text(encoding="utf-8"))
    manifest["format_version"] = 2
    current_params = manifest.get("params", {})
    exported_set = set(exported)
    allowed = [
        name
        for name in COMPONENTS
        if name in exported_set
        or name in current_params
        or any(
            name == comp and (f"{comp}.cpu" in current_params or f"{comp}.vulkan" in current_params)
            for comp in DUAL_BACKEND_COMPONENTS
        )
    ]
    params: dict[str, dict[str, str]] = {}
    for name in allowed:
        if name in DUAL_BACKEND_COMPONENTS:
            params[f"{name}.cpu"] = {
                "param": f"{name}.cpu.ncnn.param",
                "bin": f"{name}.ncnn.bin",
            }
            params[f"{name}.vulkan"] = {
                "param": f"{name}.vulkan.ncnn.param",
                "bin": f"{name}.ncnn.bin",
            }
            continue
        params[name] = {
            "param": f"{name}.ncnn.param",
            "bin": f"{name}.ncnn.bin",
        }
    manifest["params"] = params
    missing = [name for name in COMPONENTS if name not in allowed]
    setting = manifest.setdefault("setting", {})
    setting["decoder_context_length"] = decoder_len
    setting.setdefault("kv_head_cnt", 2)
    setting_fields = (
        "patch_size",
        "feat_dim",
        "latent_dim",
        "chunk_size",
        "decode_chunk_size",
        "encode_sample_rate",
        "out_sample_rate",
        "base_attn_cnt",
        "residual_attn_cnt",
        "tokens",
        "rope",
        "decoder_context_length",
        "kv_head_cnt",
    )
    manifest = {
        "model_type": "voxcpm2_tts",
        "format_version": 2,
        "params": params,
        "missing_components": missing,
        "tokenizer": {
            "type": "voxcpm2_tokenizer_json",
            "tokenizer_json": "tokenizer.json",
            "split_multichar_cjk": True,
        },
        "setting": {field: setting[field] for field in setting_fields},
    }
    model_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export VoxCPM2 component ncnn graphs for voxcpm2-ncnn.")
    parser.add_argument("--model-id", default="openbmb/VoxCPM2")
    parser.add_argument("--asset-dir", type=Path, default=Path("assets/voxcpm2"))
    parser.add_argument("--components", nargs="+", default=COMPONENTS, choices=COMPONENTS)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--update-manifest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--export-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not args.export_child:
        args.asset_dir.mkdir(parents=True, exist_ok=True)
        for name in args.components:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--model-id",
                args.model_id,
                "--asset-dir",
                str(args.asset_dir),
                "--components",
                name,
                "--no-update-manifest",
                "--export-child",
            ]
            if args.keep_intermediates:
                cmd.append("--keep-intermediates")
            cmd.append("--fp16" if args.fp16 else "--no-fp16")
            env = os.environ.copy()
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            subprocess.run(cmd, check=True, env=env)

        if args.update_manifest:
            update_manifest(args.asset_dir, list(args.components), 4)
            print(f"updated {args.asset_dir / 'model.json'}")
        else:
            print("skipped manifest update")
        return

    if len(args.components) != 1:
        raise RuntimeError("--export-child expects exactly one component")

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
        module, inputs = component_module(model, name, 4)
        inputs2 = component_shape2_inputs(model, name)
        inputshape = pnnx_inputshape(inputs)
        inputshape2 = pnnx_inputshape(inputs2) if inputs2 is not None else None
        pt_path = trace_module(name, module, inputs, args.asset_dir)
        exported.append(name)

        del inputs
        del inputs2
        del module
        del model
        del pipeline
        gc.collect()

        run_pnnx(pt_path, inputshape, args.asset_dir, args.fp16, inputshape2)
        postprocess_exported_param(name, args.asset_dir)
        if not args.keep_intermediates:
            cleanup_intermediates(name, args.asset_dir)

    if args.update_manifest:
        update_manifest(args.asset_dir, exported, 4)
        print(f"updated {args.asset_dir / 'model.json'}")
    else:
        print("skipped manifest update")


if __name__ == "__main__":
    main()
