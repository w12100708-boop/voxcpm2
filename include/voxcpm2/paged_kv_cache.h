// Paged KV cache public API for VoxCPM2-NCNN

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <vector>

#include <mat.h>

namespace voxcpm2 {

struct KvCachePair {
    ncnn::Mat key;
    ncnn::Mat value;
};

class PagedKvCache {
public:
    explicit PagedKvCache(int layer_count, int page_size = 16);

    [[nodiscard]] int layer_count() const;
    [[nodiscard]] int page_size() const;
    [[nodiscard]] int token_count() const;
    [[nodiscard]] int allocated_pages() const;

    void reset();
    [[nodiscard]] KvCachePair materialize(int layer) const;
    void append_from_contiguous(int layer, const ncnn::Mat& key, const ncnn::Mat& value);

private:
    struct Page {
        ncnn::Mat key;
        ncnn::Mat value;
        int used = 0;
    };

    struct Layer {
        std::vector<Page> pages;
    };

    void validate_layer(int layer) const;
    void ensure_shape_from(const ncnn::Mat& mat);
    [[nodiscard]] int mat_token_count(const ncnn::Mat& mat) const;
    [[nodiscard]] ncnn::Mat make_empty_cache() const;
    [[nodiscard]] ncnn::Mat make_page() const;
    void append_token(Page& page, int page_offset, const ncnn::Mat& source, int source_token, bool key);
    void copy_token(ncnn::Mat& dst, int dst_token, const ncnn::Mat& src, int src_token) const;

    std::vector<Layer> layers_;
    int page_size_ = 16;
    int token_count_ = 0;
    int width_ = 128;
    int channels_ = 2;
    bool shape_known_ = false;
};

} // namespace voxcpm2
