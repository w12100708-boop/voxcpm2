// Internal KV cache decoder runner for VoxCPM2 runtime

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include <mat.h>
#include <net.h>

namespace voxcpm2::runtime {

struct DecoderKvLayer {
    ncnn::Mat key;
    ncnn::Mat value;
};

class DecoderKvCache {
public:
    DecoderKvCache(int layer_count, int head_dim, int kv_heads);

    [[nodiscard]] int layer_count() const;
    [[nodiscard]] int token_count() const;
    [[nodiscard]] DecoderKvLayer cache_or_empty(int layer) const;

    void update(int layer, ncnn::Mat key, ncnn::Mat value);

private:
    void validate_layer(int layer) const;
    void validate_cache_mat(const ncnn::Mat& mat) const;

    std::vector<DecoderKvLayer> layers_;
    int head_dim_ = 128;
    int kv_heads_ = 2;
};

ncnn::Mat make_decoder_mask(int cur_len, int past_len);
ncnn::Mat run_decoder_with_kv(ncnn::Net& net,
                              const ncnn::Mat& embed,
                              const ncnn::Mat& mask,
                              const ncnn::Mat* cos_cache,
                              const ncnn::Mat* sin_cache,
                              DecoderKvCache& kv_cache,
                              int attn_count);

} // namespace voxcpm2::runtime
