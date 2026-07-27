// Typed schema for VoxCPM2 ncnn model manifests

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "model_manifest.h"

#include "components.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>

namespace voxcpm2::runtime {

using json = nlohmann::json;

class ManifestSchemaError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

[[noreturn]] void schema_error(std::string_view path, std::string message) {
    throw ManifestSchemaError(std::format("{}: {}", path, std::move(message)));
}

template <typename Field, std::size_t N>
void require_exact_fields(
    const json& object,
    std::string_view path,
    const std::array<Field, N>& required_fields) {
    if (not object.is_object()) {
        schema_error(path, "must be an object");
    }
    for (const Field& field : required_fields) {
        const std::string_view field_name = field;
        if (not object.contains(std::string(field_name))) {
            schema_error(path, "missing field " + std::string(field_name));
        }
    }
    for (const auto& [key, value] : object.items()) {
        static_cast<void>(value);
        if (std::ranges::none_of(required_fields, [&](const Field& field) { return key == field; })) {
            schema_error(path, "unknown field " + key);
        }
    }
}

template <typename T>
T required_value(const json& object, const char* field, std::string_view path) {
    try {
        return object.at(field).get<T>();
    } catch (const json::exception& error) {
        schema_error(std::format("{}.{}", path, field), error.what());
    }
}

void from_json(const json& object, ComponentFiles& files) {
    constexpr std::array fields{"param", "bin"};
    require_exact_fields(object, "params.<component>", fields);
    files.param = required_value<std::string>(object, "param", "params.<component>");
    files.bin = required_value<std::string>(object, "bin", "params.<component>");
}

void from_json(const json& object, TokenizerMetadata& tokenizer) {
    constexpr std::array fields{"type", "tokenizer_json", "split_multichar_cjk"};
    require_exact_fields(object, "tokenizer", fields);
    tokenizer.type = required_value<std::string>(object, "type", "tokenizer");
    tokenizer.tokenizer_json = required_value<std::string>(object, "tokenizer_json", "tokenizer");
    tokenizer.split_multichar_cjk = required_value<bool>(object, "split_multichar_cjk", "tokenizer");
}

void from_json(const json& object, TokenIds& tokens) {
    constexpr std::array fields{"audio_start", "audio_end", "ref_audio_start", "ref_audio_end"};
    require_exact_fields(object, "setting.tokens", fields);
    tokens.audio_start = required_value<int>(object, "audio_start", "setting.tokens");
    tokens.audio_end = required_value<int>(object, "audio_end", "setting.tokens");
    tokens.ref_audio_start = required_value<int>(object, "ref_audio_start", "setting.tokens");
    tokens.ref_audio_end = required_value<int>(object, "ref_audio_end", "setting.tokens");
}

void from_json(const json& object, RopeSettings& rope) {
    constexpr std::array fields{
        "type",
        "rope_head_dim",
        "rope_theta",
        "original_max_position_embeddings",
        "short_factor",
        "long_factor",
    };
    require_exact_fields(object, "setting.rope", fields);
    rope.type = required_value<std::string>(object, "type", "setting.rope");
    rope.rope_head_dim = required_value<int>(object, "rope_head_dim", "setting.rope");
    rope.rope_theta = required_value<float>(object, "rope_theta", "setting.rope");
    rope.original_max_position_embeddings =
        required_value<int>(object, "original_max_position_embeddings", "setting.rope");
    rope.short_factor = required_value<std::vector<float>>(object, "short_factor", "setting.rope");
    rope.long_factor = required_value<std::vector<float>>(object, "long_factor", "setting.rope");
}

void from_json(const json& object, ModelSettings& setting) {
    constexpr std::array fields{
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
    };
    require_exact_fields(object, "setting", fields);
    setting.patch_size = required_value<int>(object, "patch_size", "setting");
    setting.feat_dim = required_value<int>(object, "feat_dim", "setting");
    setting.latent_dim = required_value<int>(object, "latent_dim", "setting");
    setting.chunk_size = required_value<int>(object, "chunk_size", "setting");
    setting.decode_chunk_size = required_value<int>(object, "decode_chunk_size", "setting");
    setting.encode_sample_rate = required_value<int>(object, "encode_sample_rate", "setting");
    setting.out_sample_rate = required_value<int>(object, "out_sample_rate", "setting");
    setting.base_attn_cnt = required_value<int>(object, "base_attn_cnt", "setting");
    setting.residual_attn_cnt = required_value<int>(object, "residual_attn_cnt", "setting");
    setting.tokens = required_value<TokenIds>(object, "tokens", "setting");
    setting.rope = required_value<RopeSettings>(object, "rope", "setting");
    setting.decoder_context_length = required_value<int>(object, "decoder_context_length", "setting");
    setting.kv_head_cnt = required_value<int>(object, "kv_head_cnt", "setting");
}

void from_json(const json& object, ModelManifest& manifest) {
    constexpr std::array fields{
        "model_type",
        "format_version",
        "params",
        "missing_components",
        "tokenizer",
        "setting",
    };
    require_exact_fields(object, "model.json", fields);
    manifest.model_type = required_value<std::string>(object, "model_type", "model.json");
    manifest.format_version = required_value<int>(object, "format_version", "model.json");
    manifest.params = required_value<std::unordered_map<std::string, ComponentFiles>>(object, "params", "model.json");
    manifest.missing_components = required_value<std::vector<std::string>>(object, "missing_components", "model.json");
    manifest.tokenizer = required_value<TokenizerMetadata>(object, "tokenizer", "model.json");
    manifest.setting = required_value<ModelSettings>(object, "setting", "model.json");
}

void require_positive(int value, std::string_view path) {
    if (value <= 0) {
        schema_error(path, "must be positive");
    }
}

void validate_settings(const ModelSettings& setting) {
    require_positive(setting.patch_size, "setting.patch_size");
    require_positive(setting.feat_dim, "setting.feat_dim");
    require_positive(setting.latent_dim, "setting.latent_dim");
    require_positive(setting.chunk_size, "setting.chunk_size");
    require_positive(setting.decode_chunk_size, "setting.decode_chunk_size");
    require_positive(setting.encode_sample_rate, "setting.encode_sample_rate");
    require_positive(setting.out_sample_rate, "setting.out_sample_rate");
    require_positive(setting.base_attn_cnt, "setting.base_attn_cnt");
    require_positive(setting.residual_attn_cnt, "setting.residual_attn_cnt");
    require_positive(setting.decoder_context_length, "setting.decoder_context_length");
    require_positive(setting.kv_head_cnt, "setting.kv_head_cnt");

    const std::array token_ids{
        setting.tokens.audio_start,
        setting.tokens.audio_end,
        setting.tokens.ref_audio_start,
        setting.tokens.ref_audio_end,
    };
    if (std::ranges::any_of(token_ids, [](int token) { return token < 0; })) {
        schema_error("setting.tokens", "token ids must not be negative");
    }
    auto sorted_tokens = token_ids;
    std::ranges::sort(sorted_tokens);
    if (std::ranges::adjacent_find(sorted_tokens) != sorted_tokens.end()) {
        schema_error("setting.tokens", "token ids must be distinct");
    }

    const RopeSettings& rope = setting.rope;
    if (rope.type != "LongRoPE") {
        schema_error("setting.rope.type", "must be LongRoPE");
    }
    if (rope.rope_head_dim <= 0 or rope.rope_head_dim % 2 != 0) {
        schema_error("setting.rope.rope_head_dim", "must be a positive even number");
    }
    if (not std::isfinite(rope.rope_theta) or rope.rope_theta <= 0.0f) {
        schema_error("setting.rope.rope_theta", "must be finite and positive");
    }
    require_positive(
        rope.original_max_position_embeddings,
        "setting.rope.original_max_position_embeddings");
    const std::size_t factor_count = static_cast<std::size_t>(rope.rope_head_dim / 2);
    if (rope.short_factor.size() != factor_count or rope.long_factor.size() != factor_count) {
        schema_error("setting.rope", "factor lengths must equal rope_head_dim / 2");
    }
    const auto valid_factor = [](float factor) { return std::isfinite(factor) and factor > 0.0f; };
    if (not std::ranges::all_of(rope.short_factor, valid_factor) or
        not std::ranges::all_of(rope.long_factor, valid_factor)) {
        schema_error("setting.rope", "factors must be finite and positive");
    }
}

void validate_components(const ModelManifest& manifest) {
    std::unordered_set<std::string> missing;
    for (const std::string& name : manifest.missing_components) {
        if (not missing.insert(name).second) {
            schema_error("missing_components", "duplicate component " + name);
        }
    }

    std::unordered_set<std::string> allowed_param_keys;
    std::unordered_set<std::string> allowed_component_names;
    for (const ComponentSpec& spec : kComponentSpecs) {
        allowed_component_names.emplace(spec.name);
        if (spec.dual_backend) {
            allowed_param_keys.emplace(std::string(spec.name) + ".cpu");
            allowed_param_keys.emplace(std::string(spec.name) + ".vulkan");
        } else {
            allowed_param_keys.emplace(spec.name);
        }
    }
    for (const std::string& name : missing) {
        if (not allowed_component_names.contains(name)) {
            schema_error("missing_components", "unknown component " + name);
        }
    }
    for (const auto& [key, files] : manifest.params) {
        if (not allowed_param_keys.contains(key)) {
            schema_error("params", "unknown component key " + key);
        }
        if (files.param.empty() or files.bin.empty()) {
            schema_error("params." + key, "param and bin must not be empty");
        }
    }

    for (const ComponentSpec& spec : kComponentSpecs) {
        const bool declared_missing = missing.contains(std::string(spec.name));
        const auto has_key = [&](std::string suffix = {}) {
            return manifest.params.contains(std::string(spec.name) + suffix);
        };
        const bool has_params = spec.dual_backend ? has_key(".cpu") and has_key(".vulkan") : has_key();
        const bool has_partial_params = spec.dual_backend and (has_key(".cpu") != has_key(".vulkan"));
        if (has_partial_params) {
            schema_error("params", "dual-backend component is incomplete: " + std::string(spec.name));
        }
        if (declared_missing == has_params) {
            schema_error(
                "missing_components",
                declared_missing ? "component has params but is declared missing: " + std::string(spec.name)
                                 : "component has no params but is not declared missing: " + std::string(spec.name));
        }
    }
}

void validate_manifest(const ModelManifest& manifest) {
    if (manifest.model_type != "voxcpm2_tts") {
        schema_error("model_type", "must be voxcpm2_tts");
    }
    if (manifest.format_version != model_format_version) {
        schema_error("format_version", std::format("must be {}", model_format_version));
    }
    if (manifest.tokenizer.type != "voxcpm2_tokenizer_json") {
        schema_error("tokenizer.type", "must be voxcpm2_tokenizer_json");
    }
    if (manifest.tokenizer.tokenizer_json.empty()) {
        schema_error("tokenizer.tokenizer_json", "must not be empty");
    }
    if (not manifest.tokenizer.split_multichar_cjk) {
        schema_error("tokenizer.split_multichar_cjk", "must be true for this runtime");
    }
    validate_settings(manifest.setting);
    validate_components(manifest);
}

ModelManifest load_model_manifest(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (not input) {
        throw std::runtime_error("cannot open " + path.string());
    }

    try {
        json document;
        input >> document;
        ModelManifest manifest = document.get<ModelManifest>();
        validate_manifest(manifest);
        return manifest;
    } catch (const ManifestSchemaError& error) {
        throw std::runtime_error(std::format("invalid {}: {}", path.string(), error.what()));
    } catch (const json::exception& error) {
        throw std::runtime_error(std::format("invalid {}: {}", path.string(), error.what()));
    }
}

} // namespace voxcpm2::runtime
