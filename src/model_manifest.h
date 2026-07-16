// Typed schema for VoxCPM2 ncnn model manifests

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace voxcpm2::runtime {

inline constexpr int model_format_version = 2;

struct ComponentFiles {
    std::string param;
    std::string bin;
};

struct TokenizerMetadata {
    std::string type;
    std::string tokenizer_json;
    bool split_multichar_cjk = false;
};

struct TokenIds {
    int audio_start = 0;
    int audio_end = 0;
    int ref_audio_start = 0;
    int ref_audio_end = 0;
};

struct RopeSettings {
    std::string type;
    int rope_head_dim = 0;
    float rope_theta = 0.0f;
    int original_max_position_embeddings = 0;
    std::vector<float> short_factor;
    std::vector<float> long_factor;
};

struct ModelSettings {
    int patch_size = 0;
    int feat_dim = 0;
    int latent_dim = 0;
    int chunk_size = 0;
    int decode_chunk_size = 0;
    int encode_sample_rate = 0;
    int out_sample_rate = 0;
    int base_attn_cnt = 0;
    int residual_attn_cnt = 0;
    TokenIds tokens;
    RopeSettings rope;
    int decoder_context_length = 0;
    int kv_head_cnt = 0;
};

struct ModelManifest {
    std::string model_type;
    int format_version = 0;
    std::unordered_map<std::string, ComponentFiles> params;
    std::vector<std::string> missing_components;
    TokenizerMetadata tokenizer;
    ModelSettings setting;
};

[[nodiscard]] ModelManifest load_model_manifest(const std::filesystem::path& path);

} // namespace voxcpm2::runtime
