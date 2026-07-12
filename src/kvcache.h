// Internal KV cache decoder runners for VoxCPM2 runtime

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include <mat.h>
#include <net.h>

#include "profile.h"

namespace voxcpm2::runtime {

struct HostDecoderKvLayer {
    ncnn::Mat key;
    ncnn::Mat value;
};

class HostDecoderKvCache {
public:
    HostDecoderKvCache(int layer_count, int head_dim, int kv_heads);

    [[nodiscard]] int layer_count() const;
    [[nodiscard]] int token_count() const;
    [[nodiscard]] HostDecoderKvLayer cache_or_empty(int layer) const;

    void update(int layer, ncnn::Mat key, ncnn::Mat value);

private:
    void validate_layer(int layer) const;
    void validate_cache_mat(const ncnn::Mat& mat) const;

    std::vector<HostDecoderKvLayer> layers_;
    int head_dim_ = 128;
    int kv_heads_ = 2;
};

#if NCNN_VULKAN

struct VulkanDecoderKvLayer {
    ncnn::VkMat key;
    ncnn::VkMat value;
};

class VulkanDecoderKvCache {
public:
    VulkanDecoderKvCache(const ncnn::VulkanDevice* device, int layer_count, int head_dim, int kv_heads);
    ~VulkanDecoderKvCache();

    VulkanDecoderKvCache(const VulkanDecoderKvCache&) = delete;
    VulkanDecoderKvCache& operator=(const VulkanDecoderKvCache&) = delete;
    VulkanDecoderKvCache(VulkanDecoderKvCache&&) = delete;
    VulkanDecoderKvCache& operator=(VulkanDecoderKvCache&&) = delete;

    [[nodiscard]] int layer_count() const;
    [[nodiscard]] int token_count() const;
    [[nodiscard]] int head_dim() const;
    [[nodiscard]] int kv_heads() const;
    [[nodiscard]] const VulkanDecoderKvLayer& layer(int index) const;
    [[nodiscard]] const ncnn::VulkanDevice* device() const;

    void configure(ncnn::Extractor& extractor) const;
    [[nodiscard]] ncnn::Option command_option(const ncnn::Option& base) const;
    void replace(std::vector<VulkanDecoderKvLayer> layers);

private:
    void validate_layer(int layer) const;
    void validate_cache_mat(const ncnn::VkMat& mat) const;
    void release_allocators();

    const ncnn::VulkanDevice* device_ = nullptr;
    ncnn::VkAllocator* blob_allocator_ = nullptr;
    ncnn::VkAllocator* staging_allocator_ = nullptr;
    std::vector<VulkanDecoderKvLayer> layers_;
    int head_dim_ = 128;
    int kv_heads_ = 2;
};

#endif

ncnn::Mat make_decoder_mask(int cur_len, int past_len);

ncnn::Mat run_decoder_with_kv(ncnn::Net& net,
                              const ncnn::Mat& embed,
                              const ncnn::Mat& mask,
                              const ncnn::Mat* cos_cache,
                              const ncnn::Mat* sin_cache,
                              HostDecoderKvCache& kv_cache,
                              int attn_count,
                              Profile& profile);

#if NCNN_VULKAN
ncnn::Mat run_decoder_with_kv(ncnn::Net& net,
                              const ncnn::Mat& embed,
                              const ncnn::Mat& mask,
                              const ncnn::Mat* cos_cache,
                              const ncnn::Mat* sin_cache,
                              VulkanDecoderKvCache& kv_cache,
                              int attn_count,
                              Profile& profile);
#endif

} // namespace voxcpm2::runtime
