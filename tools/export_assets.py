# Prepare VoxCPM2 metadata for the NCNN runtime assets.
#
# Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
# Licensed under MIT. Not all rights reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

RUNTIME_COMPONENTS = [
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


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a VoxCPM2 ncnn asset directory.")
    parser.add_argument("--snapshot", type=Path, required=True, help="Hugging Face VoxCPM2 snapshot directory")
    parser.add_argument("--out", type=Path, default=Path("assets/voxcpm2"), help="Output asset directory")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]:
        copy_if_exists(args.snapshot / name, args.out / name)

    config_path = args.snapshot / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    rope_scaling = config["lm_config"]["rope_scaling"]

    manifest = {
        "model_type": "voxcpm2_tts",
        "format_version": 2,
        "params": {},
        "missing_components": RUNTIME_COMPONENTS,
        "tokenizer": {
            "type": "voxcpm2_tokenizer_json",
            "tokenizer_json": "tokenizer.json",
            "tokenizer_config": "tokenizer_config.json",
            "split_multichar_cjk": True,
        },
        "setting": {
            "patch_size": 4,
            "feat_dim": 64,
            "latent_dim": 64,
            "chunk_size": 640,
            "decode_chunk_size": 1920,
            "encode_sample_rate": 16000,
            "out_sample_rate": 48000,
            "base_attn_cnt": 28,
            "residual_attn_cnt": 8,
            "kv_head_cnt": config["lm_config"].get("num_key_value_heads", 2),
            "tokens": {
                "audio_start": 101,
                "audio_end": 102,
                "ref_audio_start": 103,
                "ref_audio_end": 104,
            },
            "rope": {
                "type": "LongRoPE",
                "rope_head_dim": 128,
                "rope_theta": 10000.0,
                "original_max_position_embeddings": 32768,
                "short_factor": rope_scaling["short_factor"],
                "long_factor": rope_scaling["long_factor"],
            },
        },
    }

    model_json = args.out / "model.json"
    model_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {model_json}")
    print("export ncnn components with tools/export_components.py")


if __name__ == "__main__":
    main()
