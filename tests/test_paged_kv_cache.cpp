// Paged KV cache regression tests

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "voxcpm2/paged_kv_cache.h"

#include <cmath>
#include <print>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (not condition) {
        throw std::runtime_error(message);
    }
}

ncnn::Mat make_cache(int width, int tokens, int channels, float base) {
    ncnn::Mat mat(width, tokens, channels, static_cast<std::size_t>(4));
    auto* data = static_cast<float*>(mat.data);
    for (int channel = 0; channel < channels; ++channel) {
        for (int token = 0; token < tokens; ++token) {
            for (int x = 0; x < width; ++x) {
                const std::size_t offset = (static_cast<std::size_t>(channel) * tokens + token) * width + x;
                data[offset] = base + static_cast<float>(channel * 1000 + token * 10 + x);
            }
        }
    }
    return mat;
}

float value_at(const ncnn::Mat& mat, int channel, int token, int x) {
    const auto* data = static_cast<const float*>(mat.data);
    const std::size_t offset = (static_cast<std::size_t>(channel) * mat.h + token) * mat.w + x;
    return data[offset];
}

void require_same(const ncnn::Mat& actual, const ncnn::Mat& expected) {
    require(actual.dims == expected.dims, "dims mismatch");
    require(actual.w == expected.w, "width mismatch");
    require(actual.h == expected.h, "token count mismatch");
    require(actual.c == expected.c, "channel count mismatch");
    for (int channel = 0; channel < actual.c; ++channel) {
        for (int token = 0; token < actual.h; ++token) {
            for (int x = 0; x < actual.w; ++x) {
                const float a = value_at(actual, channel, token, x);
                const float e = value_at(expected, channel, token, x);
                if (std::fabs(a - e) > 1e-6f) {
                    throw std::runtime_error("cache value mismatch");
                }
            }
        }
    }
}

} // namespace

int main() {
    {
        voxcpm2::PagedKvCache cache(2, 2);
        require(cache.layer_count() == 2, "layer count should be preserved");
        require(cache.page_size() == 2, "page size should be preserved");
        require(cache.token_count() == 0, "new cache should be empty");

        const ncnn::Mat k3 = make_cache(4, 3, 2, 10.0f);
        const ncnn::Mat v3 = make_cache(4, 3, 2, 100.0f);
        cache.append_from_contiguous(0, k3, v3);
        require(cache.token_count() == 0, "token count should advance only after final layer import");
        cache.append_from_contiguous(1, k3, v3);
        require(cache.token_count() == 3, "token count should advance after all layers are imported");
        require(cache.allocated_pages() == 4, "two layers with three tokens and page size two need four pages");

        voxcpm2::KvCachePair layer0 = cache.materialize(0);
        voxcpm2::KvCachePair layer1 = cache.materialize(1);
        require_same(layer0.key, k3);
        require_same(layer0.value, v3);
        require_same(layer1.key, k3);
        require_same(layer1.value, v3);

        const ncnn::Mat k5 = make_cache(4, 5, 2, 10.0f);
        const ncnn::Mat v5 = make_cache(4, 5, 2, 100.0f);
        cache.append_from_contiguous(0, k5, v5);
        cache.append_from_contiguous(1, k5, v5);
        require(cache.token_count() == 5, "append should import only new tokens");
        require(cache.allocated_pages() == 6, "five tokens with page size two need three pages per layer");
        require_same(cache.materialize(0).key, k5);
        require_same(cache.materialize(0).value, v5);

        cache.reset();
        require(cache.token_count() == 0, "reset should clear token count");
        require(cache.allocated_pages() == 0, "reset should release pages");
        require(cache.materialize(0).key.h == 0, "empty materialized cache should have zero tokens");
    }

    {
        voxcpm2::PagedKvCache cache(1, 4);
        const ncnn::Mat k1 = make_cache(3, 1, 2, 1.0f);
        const ncnn::Mat v1 = make_cache(3, 1, 2, 2.0f);
        cache.append_from_contiguous(0, k1, v1);
        require(cache.token_count() == 1, "single-layer cache should advance immediately");
        require_same(cache.materialize(0).key, k1);
        require_same(cache.materialize(0).value, v1);
    }

    std::println("paged kv cache tests passed");
    return 0;
}
