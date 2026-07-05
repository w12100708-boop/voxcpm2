// Paged KV cache implementation for decoder-step NCNN graphs

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "voxcpm2/paged_kv_cache.h"

#include <cstring>
#include <format>
#include <stdexcept>

namespace voxcpm2 {

PagedKvCache::PagedKvCache(int layer_count, int page_size) : layers_(static_cast<std::size_t>(layer_count)), page_size_(page_size) {
    if (layer_count <= 0) [[unlikely]] {
        throw std::runtime_error("paged kv cache layer count must be positive");
    }
    if (page_size <= 0) [[unlikely]] {
        throw std::runtime_error("paged kv cache page size must be positive");
    }
}

int PagedKvCache::layer_count() const {
    return static_cast<int>(layers_.size());
}

int PagedKvCache::page_size() const {
    return page_size_;
}

int PagedKvCache::token_count() const {
    return token_count_;
}

int PagedKvCache::allocated_pages() const {
    int total = 0;
    for (const auto& layer : layers_) {
        total += static_cast<int>(layer.pages.size());
    }
    return total;
}

void PagedKvCache::reset() {
    for (auto& layer : layers_) {
        layer.pages.clear();
    }
    token_count_ = 0;
}

KvCachePair PagedKvCache::materialize(int layer) const {
    validate_layer(layer);
    if (token_count_ == 0) {
        return KvCachePair{.key = make_empty_cache(), .value = make_empty_cache()};
    }

    ncnn::Mat key(width_, token_count_, channels_, static_cast<std::size_t>(4));
    ncnn::Mat value(width_, token_count_, channels_, static_cast<std::size_t>(4));
    const Layer& source_layer = layers_[static_cast<std::size_t>(layer)];

    int out_token = 0;
    for (const Page& page : source_layer.pages) {
        for (int page_token = 0; page_token < page.used; ++page_token) {
            copy_token(key, out_token, page.key, page_token);
            copy_token(value, out_token, page.value, page_token);
            ++out_token;
        }
    }
    if (out_token != token_count_) [[unlikely]] {
        throw std::runtime_error("paged kv cache layer token count mismatch");
    }
    return KvCachePair{.key = std::move(key), .value = std::move(value)};
}

void PagedKvCache::append_from_contiguous(int layer, const ncnn::Mat& key, const ncnn::Mat& value) {
    validate_layer(layer);
    ensure_shape_from(key);
    ensure_shape_from(value);

    const int next_token_count = mat_token_count(key);
    if (mat_token_count(value) != next_token_count) [[unlikely]] {
        throw std::runtime_error("paged kv cache key/value token counts differ");
    }
    if (next_token_count < token_count_) [[unlikely]] {
        throw std::runtime_error("paged kv cache cannot import a shorter cache");
    }

    Layer& target_layer = layers_[static_cast<std::size_t>(layer)];
    for (int source_token = token_count_; source_token < next_token_count; ++source_token) {
        if (target_layer.pages.empty() or target_layer.pages.back().used == page_size_) {
            target_layer.pages.push_back(Page{.key = make_page(), .value = make_page(), .used = 0});
        }
        Page& page = target_layer.pages.back();
        const int page_offset = page.used;
        append_token(page, page_offset, key, source_token, true);
        append_token(page, page_offset, value, source_token, false);
        ++page.used;
    }

    if (layer == layer_count() - 1) {
        token_count_ = next_token_count;
    }
}

void PagedKvCache::validate_layer(int layer) const {
    if (layer < 0 or layer >= layer_count()) [[unlikely]] {
        throw std::runtime_error(std::format("paged kv cache layer index out of range: {}", layer));
    }
}

void PagedKvCache::ensure_shape_from(const ncnn::Mat& mat) {
    if (mat.empty()) {
        return;
    }
    if (mat.elemsize != 4) [[unlikely]] {
        throw std::runtime_error("paged kv cache only supports f32 ncnn::Mat");
    }
    if (mat.dims != 3) [[unlikely]] {
        throw std::runtime_error("paged kv cache expects 3D ncnn::Mat cache tensors");
    }
    if (mat.h < 0 or mat.w <= 0 or mat.c <= 0) [[unlikely]] {
        throw std::runtime_error("paged kv cache received invalid tensor shape");
    }
    if (not shape_known_) {
        width_ = mat.w;
        channels_ = mat.c;
        shape_known_ = true;
        return;
    }
    if (mat.w != width_ or mat.c != channels_) [[unlikely]] {
        throw std::runtime_error("paged kv cache tensor shape changed");
    }
}

int PagedKvCache::mat_token_count(const ncnn::Mat& mat) const {
    if (mat.empty()) {
        return 0;
    }
    return mat.h;
}

ncnn::Mat PagedKvCache::make_empty_cache() const {
    return ncnn::Mat(width_, 0, channels_, static_cast<std::size_t>(4));
}

ncnn::Mat PagedKvCache::make_page() const {
    ncnn::Mat page(width_, page_size_, channels_, static_cast<std::size_t>(4));
    page.fill(0.0f);
    return page;
}

void PagedKvCache::append_token(Page& page, int page_offset, const ncnn::Mat& source, int source_token, bool key) {
    copy_token(key ? page.key : page.value, page_offset, source, source_token);
}

void PagedKvCache::copy_token(ncnn::Mat& dst, int dst_token, const ncnn::Mat& src, int src_token) const {
    auto* dst_data = static_cast<float*>(dst.data);
    const auto* src_data = static_cast<const float*>(src.data);
    for (int channel = 0; channel < channels_; ++channel) {
        const std::size_t dst_offset =
            (static_cast<std::size_t>(channel) * dst.h + dst_token) * width_;
        const std::size_t src_offset =
            (static_cast<std::size_t>(channel) * src.h + src_token) * width_;
        std::memcpy(dst_data + dst_offset, src_data + src_offset, static_cast<std::size_t>(width_) * sizeof(float));
    }
}

} // namespace voxcpm2
