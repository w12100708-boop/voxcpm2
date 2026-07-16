// Strict model manifest schema regression tests

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "model_manifest.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

namespace {

using json = nlohmann::json;

void require(bool condition, const char* message) {
    if (not condition) {
        throw std::runtime_error(message);
    }
}

json valid_manifest() {
    return json::parse(R"({
  "model_type": "voxcpm2_tts",
  "format_version": 2,
  "params": {
    "text_embed": {"param": "text_embed.param", "bin": "text_embed.bin"},
    "base_decoder_kv.cpu": {"param": "base_decoder_kv.cpu.param", "bin": "base_decoder_kv.bin"},
    "base_decoder_kv.vulkan": {"param": "base_decoder_kv.vulkan.param", "bin": "base_decoder_kv.bin"},
    "residual_decoder_kv.cpu": {"param": "residual_decoder_kv.cpu.param", "bin": "residual_decoder_kv.bin"},
    "residual_decoder_kv.vulkan": {"param": "residual_decoder_kv.vulkan.param", "bin": "residual_decoder_kv.bin"},
    "fsq": {"param": "fsq.param", "bin": "fsq.bin"},
    "fusion_proj": {"param": "fusion_proj.param", "bin": "fusion_proj.bin"},
    "dit_proj": {"param": "dit_proj.param", "bin": "dit_proj.bin"},
    "stop_head": {"param": "stop_head.param", "bin": "stop_head.bin"},
    "feat_encoder.cpu": {"param": "feat_encoder.cpu.param", "bin": "feat_encoder.bin"},
    "feat_encoder.vulkan": {"param": "feat_encoder.vulkan.param", "bin": "feat_encoder.bin"},
    "dit_estimator.cpu": {"param": "dit_estimator.cpu.param", "bin": "dit_estimator.bin"},
    "dit_estimator.vulkan": {"param": "dit_estimator.vulkan.param", "bin": "dit_estimator.bin"},
    "audio_vae_encoder": {"param": "audio_vae_encoder.param", "bin": "audio_vae_encoder.bin"},
    "audio_vae_decoder": {"param": "audio_vae_decoder.param", "bin": "audio_vae_decoder.bin"}
  },
  "missing_components": [],
  "tokenizer": {
    "type": "voxcpm2_tokenizer_json",
    "tokenizer_json": "tokenizer.json",
    "split_multichar_cjk": true
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
    "tokens": {
      "audio_start": 101,
      "audio_end": 102,
      "ref_audio_start": 103,
      "ref_audio_end": 104
    },
    "rope": {
      "type": "LongRoPE",
      "rope_head_dim": 4,
      "rope_theta": 10000.0,
      "original_max_position_embeddings": 32768,
      "short_factor": [1.0, 2.0],
      "long_factor": [1.0, 2.0]
    },
    "decoder_context_length": 4,
    "kv_head_cnt": 2
  }
})");
}

void write_manifest(const std::filesystem::path& path, const json& document) {
    std::ofstream output(path);
    output << document;
    if (not output) {
        throw std::runtime_error("failed to write model manifest fixture");
    }
}

void require_rejected(const std::filesystem::path& path, json document) {
    write_manifest(path, document);
    bool rejected = false;
    try {
        static_cast<void>(voxcpm2::runtime::load_model_manifest(path));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "invalid model manifest should be rejected");
}

} // namespace

int main(int argc, char** argv) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "voxcpm2_model_manifest_test.json";
    const json valid = valid_manifest();
    write_manifest(path, valid);
    const voxcpm2::runtime::ModelManifest manifest = voxcpm2::runtime::load_model_manifest(path);
    require(manifest.format_version == voxcpm2::runtime::model_format_version, "format version mismatch");
    require(manifest.setting.patch_size == 4, "patch size mismatch");
    require(manifest.params.size() == 15, "component params mismatch");

    json incomplete = valid;
    incomplete["params"] = json::object();
    incomplete["missing_components"] = json::array({
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
    });
    write_manifest(path, incomplete);
    require(
        voxcpm2::runtime::load_model_manifest(path).missing_components.size() == 11,
        "incomplete export should remain representable");

    json changed = valid;
    changed["unexpected"] = true;
    require_rejected(path, std::move(changed));

    changed = valid;
    changed["setting"].erase("patch_size");
    require_rejected(path, std::move(changed));

    changed = valid;
    changed["format_version"] = 3;
    require_rejected(path, std::move(changed));

    changed = valid;
    changed["params"].erase("base_decoder_kv.vulkan");
    require_rejected(path, std::move(changed));

    changed = valid;
    changed["params"]["unknown_component"] = {{"param", "x"}, {"bin", "y"}};
    require_rejected(path, std::move(changed));

    changed = valid;
    changed["missing_components"].push_back("text_embed");
    require_rejected(path, std::move(changed));

    changed = valid;
    changed["setting"]["rope"]["short_factor"] = json::array({1.0});
    require_rejected(path, std::move(changed));

    changed = valid;
    changed["tokenizer"]["extra"] = true;
    require_rejected(path, std::move(changed));

    std::filesystem::remove(path);
    if (argc > 1) {
        static_cast<void>(voxcpm2::runtime::load_model_manifest(argv[1]));
    }
    return 0;
}
