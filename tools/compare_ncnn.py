#!/usr/bin/env python3
"""Compare one ncnn component between precision presets from Python."""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path

import ncnn
import numpy as np


@dataclass(frozen=True)
class PrecisionConfig:
    name: str
    vulkan: bool
    fp16_storage: bool
    fp16_packed: bool
    fp16_arithmetic: bool
    packing_layout: bool


def read_component_paths(asset_dir: Path, component: str, config: PrecisionConfig) -> tuple[Path, Path]:
    manifest = json.loads((asset_dir / "model.json").read_text())
    key = component
    if key not in manifest["params"] and component in ("dit_estimator", "base_decoder_kv", "residual_decoder_kv", "feat_encoder"):
        key = f"{component}.vulkan" if config.vulkan else f"{component}.cpu"
    params = manifest["params"][key]
    return asset_dir / params["param"], asset_dir / params["bin"]


def format_layer(fields: list[str]) -> str:
    return "%-24s %-24s %s" % (fields[0], fields[1], " ".join(fields[2:]))


def add_or_replace_param(fields: list[str], key: str, value: str) -> None:
    prefix = f"{key}="
    for i, field in enumerate(fields):
        if field.startswith(prefix):
            fields[i] = f"{key}={value}"
            return
    fields.append(f"{key}={value}")


def count_header(lines: list[str]) -> tuple[str, int, int, list[str]]:
    if len(lines) < 2 or lines[0].strip() != "7767517":
        raise RuntimeError("not an ncnn param file")
    layer_count, blob_count = map(int, lines[1].split()[:2])
    return lines[0], layer_count, blob_count, lines[2:]


def rewrite_custom_sdpa_to_native(text: str) -> str:
    out: list[str] = []
    for line in text.splitlines():
        parts = line.split(maxsplit=1)
        if parts and parts[0] == "VoxCPM2SDPA":
            line = "SDPA" + line[len("VoxCPM2SDPA") :]
        out.append(line)
    return "\n".join(out) + "\n"


def rewrite_dtype_adapter_to_rmsnorm(text: str) -> str:
    """Collapse VoxCPM2DTypeAdapter -> RMSNorm -> Cast triples back to plain RMSNorm.

    This allows the Python comparison harness to load the Vulkan param without
    the C++-only VoxCPM2DTypeAdapter custom layer.
    """
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    magic = lines[0]
    body = lines[2:]
    out: list[str] = []
    i = 0
    while i < len(body):
        fields = body[i].split()
        if (
            fields
            and fields[0] == "VoxCPM2DTypeAdapter"
            and i + 2 < len(body)
        ):
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
                out.append(format_layer(rms_fields))
                i += 3
                continue
        out.append(body[i])
        i += 1

    layer_lines = [line for line in out if line.strip()]
    layer_count = len(layer_lines)
    blob_count = sum(int(line.split()[3]) for line in layer_lines)
    return "\n".join([magic, f"{layer_count} {blob_count}", *layer_lines]) + "\n"


def apply_param_rewrite(text: str, rewrite: str) -> str:
    if rewrite == "none":
        return text
    if rewrite == "gemm-output-fp32":
        out: list[str] = []
        fixed = 0
        for line in text.splitlines():
            fields = line.split()
            if fields and fields[0] == "Gemm":
                add_or_replace_param(fields, "13", "1")
                line = format_layer(fields)
                fixed += 1
            out.append(line)
        if fixed == 0:
            raise RuntimeError("gemm-output-fp32 matched no Gemm layers")
        return "\n".join(out) + "\n"
    if rewrite in {"sdpa-boundary-fp32", "rmsnorm-boundary-fp32"}:
        magic, _layer_count, blob_count, body = count_header(text.splitlines())
        target_layers = {"SDPA", "VoxCPM2SDPA"} if rewrite == "sdpa-boundary-fp32" else {"RMSNorm"}
        out_body: list[str] = []
        inserted = 0
        for line in body:
            fields = line.split()
            if not fields or fields[0] not in target_layers:
                out_body.append(line)
                continue
            bottom_count = int(fields[2])
            top_count = int(fields[3])
            bottoms = fields[4 : 4 + bottom_count]
            tops = fields[4 + bottom_count : 4 + bottom_count + top_count]
            params = fields[4 + bottom_count + top_count :]

            cast_bottoms: list[str] = []
            for i, bottom in enumerate(bottoms):
                cast_top = f"{fields[1]}_fp32_in{i}"
                out_body.append(format_layer(["Cast", f"{fields[1]}_cast_in{i}", "1", "1", bottom, cast_top, "0=0", "1=1"]))
                cast_bottoms.append(cast_top)
                inserted += 1

            internal_tops = [f"{fields[1]}_fp32_out{i}" for i in range(top_count)]
            layer_fields = fields[:4] + cast_bottoms + internal_tops + params
            out_body.append(format_layer(layer_fields))
            for i, top in enumerate(tops):
                out_body.append(format_layer(["Cast", f"{fields[1]}_cast_out{i}", "1", "1", internal_tops[i], top, "0=0", "1=1"]))
                inserted += 1

        if inserted == 0:
            raise RuntimeError(f"{rewrite} matched no layers")
        layer_count = len([line for line in out_body if line.strip()])
        return "\n".join([magic, f"{layer_count} {blob_count + inserted}", *out_body]) + "\n"
    raise RuntimeError(f"unknown param rewrite: {rewrite}")


def parse_top_blobs(param_text: str) -> list[tuple[str, str, str]]:
    blobs: list[str] = []
    layers: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for line in param_text.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0].isdigit():
            continue
        try:
            bottom_count = int(parts[2])
            top_count = int(parts[3])
        except ValueError:
            continue
        tops = parts[4 + bottom_count : 4 + bottom_count + top_count]
        for blob in tops:
            if blob not in seen:
                seen.add(blob)
                blobs.append(blob)
                layers.append((blob, parts[0], parts[1]))
    return layers


def precision_preset(name: str) -> PrecisionConfig:
    presets = {
        "safe-vk": PrecisionConfig(name, True, False, False, True, True),
        "full-fp16-vk": PrecisionConfig(name, True, True, True, True, True),
        "fp16-storage-only-vk": PrecisionConfig(name, True, True, False, True, True),
        "fp16-packed-only-vk": PrecisionConfig(name, True, False, True, True, True),
        "no-fp16-arithmetic-vk": PrecisionConfig(name, True, True, True, False, True),
        "no-packing-layout-vk": PrecisionConfig(name, True, True, True, True, False),
        "cpu": PrecisionConfig(name, False, True, True, True, True),
    }
    try:
        return presets[name]
    except KeyError as exc:
        raise RuntimeError(f"unknown precision preset: {name}") from exc


def make_option(config: PrecisionConfig, args: argparse.Namespace) -> ncnn.Option:
    opt = ncnn.Option()
    opt.num_threads = args.threads
    opt.use_vulkan_compute = config.vulkan
    opt.use_fp16_storage = config.fp16_storage
    opt.use_fp16_packed = config.fp16_packed
    opt.use_fp16_arithmetic = config.fp16_arithmetic
    opt.use_bf16_storage = False
    opt.use_bf16_packed = False
    opt.use_int8_inference = False
    opt.use_int8_storage = False
    opt.use_int8_packed = False
    opt.use_int8_arithmetic = False
    opt.use_packing_layout = config.packing_layout
    return opt


def load_net(param_text: str, bin_path: Path, config: PrecisionConfig, args: argparse.Namespace) -> ncnn.Net:
    net = ncnn.Net()
    net.opt = make_option(config, args)
    if config.vulkan:
        net.set_vulkan_device(args.vulkan_device)
    if net.load_param_mem(param_text) != 0:
        raise RuntimeError("load_param_mem failed")
    if net.load_model(str(bin_path)) != 0:
        raise RuntimeError(f"load_model failed: {bin_path}")
    return net


def default_inputs(component: str) -> dict[str, np.ndarray]:
    def seq(shape: tuple[int, ...], scale: float, offset: float = 0.0) -> np.ndarray:
        return np.sin(np.arange(np.prod(shape), dtype=np.float32) * scale + offset).reshape(shape)

    if component == "fsq":
        return {"in0": seq((1, 2048), 0.003)}
    if component == "text_embed":
        return {"in0": np.array([1, 2, 3, 4], dtype=np.int32)}
    if component in {"fusion_proj", "dit_proj"}:
        return {"in0": seq((1, 2048), 0.003), "in1": seq((1, 2048), 0.005, 0.2)}
    if component == "stop_head":
        return {"in0": seq((1, 2048), 0.003)}
    if component == "feat_encoder":
        return {"in0": seq((4, 64), 0.017, 0.4)}
    if component == "audio_vae_encoder":
        return {"in0": seq((1, 5120), 0.01)}
    if component == "audio_vae_decoder":
        return {"in0": seq((64, 8), 0.017, 0.2)}
    if component == "dit_estimator":
        return {
            "in0": seq((2, 64, 4), 0.013),
            "in1": seq((2, 2048), 0.003, 0.2),
            "in2": np.array([0.70710677, 0.70710677], dtype=np.float32),
            "in3": seq((2, 64, 4), 0.017, 0.4),
            "in4": np.array([0.0, 0.0], dtype=np.float32),
        }
    if component == "base_decoder_kv":
        inputs = {
            "in0": seq((1, 2048), 0.003),
            "in1": np.array([[0.0, 0.0]], dtype=np.float32),
            "in2": seq((1, 128), 0.011),
            "in3": seq((1, 128), 0.013, 0.2),
        }
        for i in range(28):
            inputs[f"cache_k{i}"] = seq((2, 1, 128), 0.001 + i * 0.0001, 0.1)
            inputs[f"cache_v{i}"] = seq((2, 1, 128), 0.0013 + i * 0.0001, 0.2)
        return inputs
    if component == "residual_decoder_kv":
        inputs = {
            "in0": seq((1, 2048), 0.003),
            "in1": np.array([[0.0, 0.0]], dtype=np.float32),
        }
        for i in range(8):
            inputs[f"cache_k{i}"] = seq((2, 1, 128), 0.001 + i * 0.0001, 0.1)
            inputs[f"cache_v{i}"] = seq((2, 1, 128), 0.0013 + i * 0.0001, 0.2)
        return inputs
    raise RuntimeError(f"{component}: provide --inputs npz or add a default fixture")


def load_inputs(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.inputs:
        data = np.load(args.inputs)
        return {name: np.asarray(data[name]) for name in data.files}
    return default_inputs(args.component)


def mat_from_numpy(array: np.ndarray) -> ncnn.Mat:
    if array.dtype == np.int32:
        return ncnn.Mat(np.ascontiguousarray(array))
    return ncnn.Mat(np.ascontiguousarray(array, dtype=np.float32))


def extract(net: ncnn.Net, inputs: dict[str, np.ndarray], blob: str) -> np.ndarray | None:
    ex = net.create_extractor()
    ex.set_light_mode(False)
    for name, value in inputs.items():
        ex.input(name, mat_from_numpy(value))
    ret, mat = ex.extract(blob)
    if ret != 0 or mat.empty():
        return None
    return np.asarray(mat.numpy(), dtype=np.float32).copy()


def diff(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if a.shape != b.shape:
        return {"shape_mismatch": 1.0}
    aa = a.reshape(-1)
    bb = b.reshape(-1)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if not np.any(mask):
        return {"finite": 0.0}
    d = np.abs(aa[mask] - bb[mask])
    rel = d / np.maximum(np.abs(bb[mask]), 1e-6)
    return {
        "finite": float(mask.sum()),
        "nonfinite_a": float(aa.size - np.isfinite(aa).sum()),
        "nonfinite_b": float(bb.size - np.isfinite(bb).sum()),
        "zero_a": float((aa == 0).sum()),
        "zero_b": float((bb == 0).sum()),
        "max_abs": float(d.max(initial=0.0)),
        "mean_abs": float(d.mean()),
        "rms_abs": float(math.sqrt(float(np.mean(d * d)))),
        "mean_rel": float(rel.mean()),
    }


def fails_threshold(stats: dict[str, float], max_abs: float, mean_abs: float) -> bool:
    return bool(
        stats.get("shape_mismatch", 0.0)
        or stats.get("nonfinite_a", 0.0)
        or stats.get("nonfinite_b", 0.0)
        or stats.get("max_abs", 0.0) >= max_abs
        or stats.get("mean_abs", 0.0) >= mean_abs
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, default=Path("build/voxcpm2_fp16_assets"))
    parser.add_argument("--component", default="dit_estimator")
    parser.add_argument("--param", type=Path)
    parser.add_argument("--bin", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--blob", action="append")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-abs-threshold", type=float, default=1e-3)
    parser.add_argument("--mean-abs-threshold", type=float, default=1e-4)
    parser.add_argument("--base-preset", default="safe-vk")
    parser.add_argument("--target-preset", default="full-fp16-vk")
    parser.add_argument("--rewrite", choices=["none", "gemm-output-fp32", "sdpa-boundary-fp32", "rmsnorm-boundary-fp32"], default="none")
    parser.add_argument("--native-sdpa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-rmsnorm", action=argparse.BooleanOptionalAction, default=True, help="collapse VoxCPM2DTypeAdapter triples to plain RMSNorm")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--vulkan-device", type=int, default=0)
    args = parser.parse_args()

    inputs = load_inputs(args)
    base_config = precision_preset(args.base_preset)
    target_config = precision_preset(args.target_preset)
    if args.param:
        base_param_path = target_param_path = args.param
        bin_path = args.bin
        if bin_path is None:
            _, bin_path = read_component_paths(args.asset_dir, args.component, target_config)
    else:
        base_param_path, base_bin_path = read_component_paths(args.asset_dir, args.component, base_config)
        target_param_path, target_bin_path = read_component_paths(args.asset_dir, args.component, target_config)
        if base_bin_path != target_bin_path and args.bin is None:
            raise RuntimeError(f"base and target bins differ: {base_bin_path} vs {target_bin_path}")
        bin_path = args.bin or target_bin_path
    base_param_text = apply_param_rewrite(base_param_path.read_text(), args.rewrite)
    target_param_text = apply_param_rewrite(target_param_path.read_text(), args.rewrite)
    if args.native_sdpa:
        base_param_text = rewrite_custom_sdpa_to_native(base_param_text)
        target_param_text = rewrite_custom_sdpa_to_native(target_param_text)
    if args.native_rmsnorm:
        base_param_text = rewrite_dtype_adapter_to_rmsnorm(base_param_text)
        target_param_text = rewrite_dtype_adapter_to_rmsnorm(target_param_text)

    ncnn.create_gpu_instance()
    base = None
    target = None
    try:
        base = load_net(base_param_text, bin_path, base_config, args)
        target = load_net(target_param_text, bin_path, target_config, args)
        blobs = parse_top_blobs(target_param_text) if args.scan else [(blob, "", "") for blob in (args.blob or ["out0"])]
        printed = 0
        for blob, layer_type, layer_name in blobs:
            base_out = extract(base, inputs, blob)
            target_out = extract(target, inputs, blob)
            if base_out is None or target_out is None:
                continue
            stats = diff(target_out, base_out)
            if args.scan and not fails_threshold(stats, args.max_abs_threshold, args.mean_abs_threshold):
                continue
            layer = f" {layer_type}/{layer_name}" if layer_name else ""
            print(f"{blob}:{layer} base_shape={base_out.shape} target_shape={target_out.shape} {stats}")
            if base_out.size <= 8:
                print(f"  base  ={base_out.reshape(-1).tolist()}")
                print(f"  target={target_out.reshape(-1).tolist()}")
            printed += 1
            if args.scan and printed >= args.limit:
                break
    finally:
        base = None
        target = None
        gc.collect()
        ncnn.destroy_gpu_instance()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
