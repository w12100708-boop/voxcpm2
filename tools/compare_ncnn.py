#!/usr/bin/env python3
"""Compare one ncnn component on CPU and Vulkan from Python."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import ncnn
import numpy as np


def read_component_paths(asset_dir: Path, component: str) -> tuple[Path, Path]:
    manifest = json.loads((asset_dir / "model.json").read_text())
    params = manifest["params"][component]
    return asset_dir / params["param"], asset_dir / params["bin"]


def rewrite_param_text(text: str, native_sdpa: bool) -> str:
    if not native_sdpa:
        return text
    out: list[str] = []
    for line in text.splitlines():
        parts = line.split(maxsplit=1)
        if parts and parts[0] == "VoxCPM2SDPA":
            line = "SDPA" + line[len("VoxCPM2SDPA") :]
        out.append(line)
    return "\n".join(out) + "\n"


def parse_top_blobs(param_text: str) -> list[str]:
    blobs: list[str] = []
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
    return blobs


def make_option(vulkan: bool, args: argparse.Namespace) -> ncnn.Option:
    opt = ncnn.Option()
    opt.num_threads = args.threads
    opt.use_vulkan_compute = vulkan
    opt.use_fp16_storage = args.fp16_storage
    opt.use_fp16_packed = args.fp16_packed
    opt.use_fp16_arithmetic = args.fp16_arithmetic
    opt.use_bf16_storage = False
    opt.use_bf16_packed = False
    opt.use_int8_inference = False
    opt.use_int8_storage = False
    opt.use_int8_packed = False
    opt.use_int8_arithmetic = False
    opt.use_packing_layout = args.packing_layout
    return opt


def load_net(param_text: str, bin_path: Path, vulkan: bool, args: argparse.Namespace) -> ncnn.Net:
    net = ncnn.Net()
    net.opt = make_option(vulkan, args)
    if vulkan:
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
        "max_abs": float(d.max(initial=0.0)),
        "mean_abs": float(d.mean()),
        "rms_abs": float(math.sqrt(float(np.mean(d * d)))),
        "mean_rel": float(rel.mean()),
    }


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
    parser.add_argument("--native-sdpa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16-storage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16-packed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16-arithmetic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--packing-layout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--vulkan-device", type=int, default=0)
    args = parser.parse_args()

    param_path, bin_path = (args.param, args.bin) if args.param and args.bin else read_component_paths(args.asset_dir, args.component)
    raw_param = param_path.read_text()
    param_text = rewrite_param_text(raw_param, args.native_sdpa)
    inputs = load_inputs(args)

    ncnn.create_gpu_instance()
    cpu = None
    vk = None
    try:
        cpu = load_net(param_text, bin_path, False, args)
        vk = load_net(param_text, bin_path, True, args)
        blobs = parse_top_blobs(param_text) if args.scan else (args.blob or ["out0"])
        printed = 0
        for blob in blobs:
            cpu_out = extract(cpu, inputs, blob)
            vk_out = extract(vk, inputs, blob)
            if cpu_out is None or vk_out is None:
                continue
            stats = diff(vk_out, cpu_out)
            if args.scan and stats.get("max_abs", 0.0) < 1e-3 and stats.get("mean_abs", 0.0) < 1e-4:
                continue
            print(f"{blob}: cpu_shape={cpu_out.shape} vk_shape={vk_out.shape} {stats}")
            if cpu_out.size <= 8:
                print(f"  cpu={cpu_out.reshape(-1).tolist()}")
                print(f"  vk ={vk_out.reshape(-1).tolist()}")
            printed += 1
            if args.scan and printed >= args.limit:
                break
    finally:
        cpu = None
        vk = None
        gc.collect()
        ncnn.destroy_gpu_instance()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
