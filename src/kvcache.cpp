// Internal KV cache decoder runner for VoxCPM2 runtime

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "kvcache.h"

#include <format>
#include <stdexcept>
#include <utility>

namespace voxcpm2::runtime {

DecoderKvCache::DecoderKvCache(int layer_count, int head_dim, int kv_heads)
    : layers_(static_cast<std::size_t>(layer_count)), head_dim_(head_dim), kv_heads_(kv_heads) {
    if (layer_count <= 0) [[unlikely]] {
        throw std::runtime_error("decoder kv layer count must be positive");
    }
    if (head_dim <= 0 or kv_heads <= 0) [[unlikely]] {
        throw std::runtime_error("decoder kv shape must be positive");
    }
}

int DecoderKvCache::layer_count() const {
    return static_cast<int>(layers_.size());
}

int DecoderKvCache::token_count() const {
    if (layers_.empty() or layers_.front().key.empty()) {
        return 0;
    }
    return layers_.front().key.h;
}

DecoderKvLayer DecoderKvCache::cache_or_empty(int layer) const {
    validate_layer(layer);
    const DecoderKvLayer& existing = layers_[static_cast<std::size_t>(layer)];
    if (existing.key.empty() and existing.value.empty()) {
        return DecoderKvLayer{
            .key = ncnn::Mat(head_dim_, 0, kv_heads_, static_cast<std::size_t>(4)),
            .value = ncnn::Mat(head_dim_, 0, kv_heads_, static_cast<std::size_t>(4)),
        };
    }
    return existing;
}

void DecoderKvCache::update(int layer, ncnn::Mat key, ncnn::Mat value) {
    validate_layer(layer);
    validate_cache_mat(key);
    validate_cache_mat(value);
    if (key.h != value.h) [[unlikely]] {
        throw std::runtime_error("decoder kv key/value token counts differ");
    }
    layers_[static_cast<std::size_t>(layer)] = DecoderKvLayer{.key = std::move(key), .value = std::move(value)};
}

void DecoderKvCache::validate_layer(int layer) const {
    if (layer < 0 or layer >= layer_count()) [[unlikely]] {
        throw std::runtime_error(std::format("decoder kv layer index out of range: {}", layer));
    }
}

void DecoderKvCache::validate_cache_mat(const ncnn::Mat& mat) const {
    if (mat.empty()) [[unlikely]] {
        throw std::runtime_error("decoder kv cache output is empty");
    }
    if (mat.elemsize != 4 or mat.dims != 3 or mat.w != head_dim_ or mat.c != kv_heads_) [[unlikely]] {
        throw std::runtime_error("decoder kv cache output shape mismatch");
    }
}

ncnn::Mat make_decoder_mask(int cur_len, int past_len) {
    if (cur_len <= 0 or past_len < 0) [[unlikely]] {
        throw std::runtime_error("invalid decoder attention mask shape");
    }
    const int dst_len = past_len + cur_len;
    ncnn::Mat mat(dst_len, cur_len);
    mat.fill(0.0f);
    auto* data = static_cast<float*>(mat.data);
    for (int row = 0; row < cur_len; ++row) {
        for (int col = past_len + row + 1; col < dst_len; ++col) {
            data[static_cast<std::size_t>(row) * dst_len + col] = -10000.0f;
        }
    }
    return mat;
}

ncnn::Mat run_decoder_with_kv(ncnn::Net& net,
                              const ncnn::Mat& embed,
                              const ncnn::Mat& mask,
                              const ncnn::Mat* cos_cache,
                              const ncnn::Mat* sin_cache,
                              DecoderKvCache& kv_cache,
                              int attn_count) {
    if (kv_cache.layer_count() != attn_count) [[unlikely]] {
        throw std::runtime_error("decoder kv layer count mismatch");
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", embed);
    ex.input("in1", mask);
    if (cos_cache != nullptr and sin_cache != nullptr) {
        ex.input("in2", *cos_cache);
        ex.input("in3", *sin_cache);
    }

    std::vector<DecoderKvLayer> input_cache;
    input_cache.reserve(static_cast<std::size_t>(attn_count));
    for (int i = 0; i < attn_count; ++i) {
        input_cache.push_back(kv_cache.cache_or_empty(i));
        const std::string k_name = std::format("cache_k{}", i);
        const std::string v_name = std::format("cache_v{}", i);
        ex.input(k_name.c_str(), input_cache.back().key);
        ex.input(v_name.c_str(), input_cache.back().value);
    }

    for (int i = 0; i < attn_count; ++i) {
        const std::string k_name = std::format("out_cache_k{}", i);
        const std::string v_name = std::format("out_cache_v{}", i);
        ncnn::Mat k_cache;
        ncnn::Mat v_cache;
        if (ex.extract(k_name.c_str(), k_cache) != 0 or k_cache.empty() or
            ex.extract(v_name.c_str(), v_cache) != 0 or v_cache.empty()) [[unlikely]] {
            throw std::runtime_error("failed to extract decoder kv cache");
        }
        kv_cache.update(i, std::move(k_cache), std::move(v_cache));
    }

    ncnn::Mat out;
    if (ex.extract("out0", out) != 0 or out.empty()) [[unlikely]] {
        throw std::runtime_error("failed to extract decoder out0");
    }
    return out;
}

} // namespace voxcpm2::runtime
