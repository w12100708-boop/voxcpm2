// Internal KV cache decoder runners for VoxCPM2 runtime

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "kvcache.h"

#include <format>
#include <stdexcept>
#include <utility>

#if NCNN_VULKAN
#include <command.h>
#include <gpu.h>
#endif

namespace voxcpm2::runtime {
namespace {

void check_cache_shape(int dims, int w, int c, int h, int head_dim, int kv_heads) {
    if (dims != 3 or w != head_dim or c != kv_heads or h <= 0) [[unlikely]] {
        throw std::runtime_error("decoder kv cache output shape mismatch");
    }
}

void check_input(int result, std::string_view name) {
    if (result != 0) [[unlikely]] {
        throw std::runtime_error("failed to bind decoder input " + std::string(name));
    }
}

std::string cache_name(std::string_view prefix, std::string_view kind, int layer) {
    return std::format("{}{}{}", prefix, kind, layer);
}

} // namespace

HostDecoderKvCache::HostDecoderKvCache(int layer_count, int head_dim, int kv_heads)
    : layers_(static_cast<std::size_t>(layer_count)), head_dim_(head_dim), kv_heads_(kv_heads) {
    if (layer_count <= 0) [[unlikely]] {
        throw std::runtime_error("decoder kv layer count must be positive");
    }
    if (head_dim <= 0 or kv_heads <= 0) [[unlikely]] {
        throw std::runtime_error("decoder kv shape must be positive");
    }
}

int HostDecoderKvCache::layer_count() const {
    return static_cast<int>(layers_.size());
}

int HostDecoderKvCache::token_count() const {
    if (layers_.empty() or layers_.front().key.empty()) {
        return 0;
    }
    return layers_.front().key.h;
}

HostDecoderKvLayer HostDecoderKvCache::cache_or_empty(int layer) const {
    validate_layer(layer);
    const HostDecoderKvLayer& existing = layers_[static_cast<std::size_t>(layer)];
    if (existing.key.empty() and existing.value.empty()) {
        return HostDecoderKvLayer{
            .key = ncnn::Mat(head_dim_, 0, kv_heads_, static_cast<std::size_t>(4)),
            .value = ncnn::Mat(head_dim_, 0, kv_heads_, static_cast<std::size_t>(4)),
        };
    }
    return existing;
}

void HostDecoderKvCache::update(int layer, ncnn::Mat key, ncnn::Mat value) {
    validate_layer(layer);
    validate_cache_mat(key);
    validate_cache_mat(value);
    if (key.h != value.h) [[unlikely]] {
        throw std::runtime_error("decoder kv key/value token counts differ");
    }
    layers_[static_cast<std::size_t>(layer)] = HostDecoderKvLayer{.key = std::move(key), .value = std::move(value)};
}

void HostDecoderKvCache::validate_layer(int layer) const {
    if (layer < 0 or layer >= layer_count()) [[unlikely]] {
        throw std::runtime_error(std::format("decoder kv layer index out of range: {}", layer));
    }
}

void HostDecoderKvCache::validate_cache_mat(const ncnn::Mat& mat) const {
    if (mat.empty() or mat.elemsize != 4) [[unlikely]] {
        throw std::runtime_error("decoder host kv cache output dtype mismatch");
    }
    check_cache_shape(mat.dims, mat.w, mat.c, mat.h, head_dim_, kv_heads_);
}

#if NCNN_VULKAN

VulkanDecoderKvCache::VulkanDecoderKvCache(
    const ncnn::VulkanDevice* device,
    int layer_count,
    int head_dim,
    int kv_heads)
    : device_(device), layers_(static_cast<std::size_t>(layer_count)), head_dim_(head_dim), kv_heads_(kv_heads) {
    if (device_ == nullptr) [[unlikely]] {
        throw std::runtime_error("decoder Vulkan cache requires a Vulkan device");
    }
    if (layer_count <= 0) [[unlikely]] {
        throw std::runtime_error("decoder kv layer count must be positive");
    }
    if (head_dim <= 0 or kv_heads <= 0) [[unlikely]] {
        throw std::runtime_error("decoder kv shape must be positive");
    }
    blob_allocator_ = device_->acquire_blob_allocator();
    staging_allocator_ = device_->acquire_staging_allocator();
    if (blob_allocator_ == nullptr or staging_allocator_ == nullptr) [[unlikely]] {
        release_allocators();
        throw std::runtime_error("failed to acquire decoder Vulkan allocators");
    }
}

VulkanDecoderKvCache::~VulkanDecoderKvCache() {
    layers_.clear();
    release_allocators();
}

int VulkanDecoderKvCache::layer_count() const {
    return static_cast<int>(layers_.size());
}

int VulkanDecoderKvCache::token_count() const {
    if (layers_.empty() or layers_.front().key.empty()) {
        return 0;
    }
    return layers_.front().key.h;
}

int VulkanDecoderKvCache::head_dim() const {
    return head_dim_;
}

int VulkanDecoderKvCache::kv_heads() const {
    return kv_heads_;
}

const VulkanDecoderKvLayer& VulkanDecoderKvCache::layer(int index) const {
    validate_layer(index);
    return layers_[static_cast<std::size_t>(index)];
}

const ncnn::VulkanDevice* VulkanDecoderKvCache::device() const {
    return device_;
}

void VulkanDecoderKvCache::configure(ncnn::Extractor& extractor) const {
    extractor.set_blob_vkallocator(blob_allocator_);
    extractor.set_workspace_vkallocator(blob_allocator_);
    extractor.set_staging_vkallocator(staging_allocator_);
}

ncnn::Option VulkanDecoderKvCache::command_option(const ncnn::Option& base) const {
    ncnn::Option option = base;
    option.blob_vkallocator = blob_allocator_;
    option.workspace_vkallocator = blob_allocator_;
    option.staging_vkallocator = staging_allocator_;
    return option;
}

void VulkanDecoderKvCache::replace(std::vector<VulkanDecoderKvLayer> layers) {
    if (layers.size() != layers_.size()) [[unlikely]] {
        throw std::runtime_error("decoder Vulkan cache layer count mismatch");
    }
    int expected_tokens = -1;
    for (const VulkanDecoderKvLayer& layer : layers) {
        validate_cache_mat(layer.key);
        validate_cache_mat(layer.value);
        if (layer.key.h != layer.value.h) [[unlikely]] {
            throw std::runtime_error("decoder Vulkan key/value token counts differ");
        }
        if (expected_tokens < 0) {
            expected_tokens = layer.key.h;
        } else if (layer.key.h != expected_tokens) [[unlikely]] {
            throw std::runtime_error("decoder Vulkan layers have inconsistent token counts");
        }
    }
    layers_ = std::move(layers);
}

void VulkanDecoderKvCache::validate_layer(int layer) const {
    if (layer < 0 or layer >= layer_count()) [[unlikely]] {
        throw std::runtime_error(std::format("decoder Vulkan kv layer index out of range: {}", layer));
    }
}

void VulkanDecoderKvCache::validate_cache_mat(const ncnn::VkMat& mat) const {
    if (mat.empty() or (mat.elembits() != 16 and mat.elembits() != 32)) [[unlikely]] {
        throw std::runtime_error("decoder Vulkan kv cache output dtype mismatch");
    }
    check_cache_shape(mat.dims, mat.w, mat.c, mat.h, head_dim_, kv_heads_);
}

void VulkanDecoderKvCache::release_allocators() {
    if (device_ == nullptr) {
        return;
    }
    if (blob_allocator_ != nullptr) {
        device_->reclaim_blob_allocator(blob_allocator_);
        blob_allocator_ = nullptr;
    }
    if (staging_allocator_ != nullptr) {
        device_->reclaim_staging_allocator(staging_allocator_);
        staging_allocator_ = nullptr;
    }
}

#endif

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
                              HostDecoderKvCache& kv_cache,
                              int attn_count,
                              Profile&) {
    if (kv_cache.layer_count() != attn_count) [[unlikely]] {
        throw std::runtime_error("decoder kv layer count mismatch");
    }

    ncnn::Extractor ex = net.create_extractor();
    check_input(ex.input("in0", embed), "in0");
    check_input(ex.input("in1", mask), "in1");
    if (cos_cache != nullptr and sin_cache != nullptr) {
        check_input(ex.input("in2", *cos_cache), "in2");
        check_input(ex.input("in3", *sin_cache), "in3");
    }

    std::vector<HostDecoderKvLayer> input_cache;
    input_cache.reserve(static_cast<std::size_t>(attn_count));
    for (int i = 0; i < attn_count; ++i) {
        input_cache.push_back(kv_cache.cache_or_empty(i));
        const std::string k_name = cache_name("cache_", "k", i);
        const std::string v_name = cache_name("cache_", "v", i);
        check_input(ex.input(k_name.c_str(), input_cache.back().key), k_name);
        check_input(ex.input(v_name.c_str(), input_cache.back().value), v_name);
    }

    for (int i = 0; i < attn_count; ++i) {
        const std::string k_name = cache_name("out_cache_", "k", i);
        const std::string v_name = cache_name("out_cache_", "v", i);
        ncnn::Mat k_cache;
        ncnn::Mat v_cache;
        if (ex.extract(k_name.c_str(), k_cache) != 0 or k_cache.empty() or
            ex.extract(v_name.c_str(), v_cache) != 0 or v_cache.empty()) [[unlikely]] {
            throw std::runtime_error("failed to extract decoder host kv cache");
        }
        kv_cache.update(i, std::move(k_cache), std::move(v_cache));
    }

    ncnn::Mat out;
    if (ex.extract("out0", out) != 0 or out.empty()) [[unlikely]] {
        throw std::runtime_error("failed to extract decoder out0");
    }
    return out;
}

#if NCNN_VULKAN
ncnn::Mat run_decoder_with_kv(ncnn::Net& net,
                              const ncnn::Mat& embed,
                              const ncnn::Mat& mask,
                              const ncnn::Mat* cos_cache,
                              const ncnn::Mat* sin_cache,
                              VulkanDecoderKvCache& kv_cache,
                              int attn_count,
                              Profile& profile) {
    if (kv_cache.layer_count() != attn_count) [[unlikely]] {
        throw std::runtime_error("decoder Vulkan kv layer count mismatch");
    }
    if (net.vulkan_device() != kv_cache.device()) [[unlikely]] {
        throw std::runtime_error("decoder Vulkan cache device mismatch");
    }

    ncnn::Extractor ex = net.create_extractor();
    kv_cache.configure(ex);
    check_input(ex.input("in0", embed), "in0");
    check_input(ex.input("in1", mask), "in1");
    if (cos_cache != nullptr and sin_cache != nullptr) {
        check_input(ex.input("in2", *cos_cache), "in2");
        check_input(ex.input("in3", *sin_cache), "in3");
    }

    std::vector<HostDecoderKvLayer> empty_inputs;
    empty_inputs.reserve(static_cast<std::size_t>(attn_count));
    for (int i = 0; i < attn_count; ++i) {
        const std::string k_name = cache_name("cache_", "k", i);
        const std::string v_name = cache_name("cache_", "v", i);
        const VulkanDecoderKvLayer& existing = kv_cache.layer(i);
        if (existing.key.empty() and existing.value.empty()) {
            empty_inputs.push_back(HostDecoderKvLayer{
                .key = ncnn::Mat(kv_cache.head_dim(), 0, kv_cache.kv_heads(), static_cast<std::size_t>(4)),
                .value = ncnn::Mat(kv_cache.head_dim(), 0, kv_cache.kv_heads(), static_cast<std::size_t>(4)),
            });
            check_input(ex.input(k_name.c_str(), empty_inputs.back().key), k_name);
            check_input(ex.input(v_name.c_str(), empty_inputs.back().value), v_name);
        } else {
            check_input(ex.input(k_name.c_str(), existing.key), k_name);
            check_input(ex.input(v_name.c_str(), existing.value), v_name);
        }
    }

    ncnn::VkCompute command(kv_cache.device());
    std::vector<VulkanDecoderKvLayer> next_cache(static_cast<std::size_t>(attn_count));
    for (int i = 0; i < attn_count; ++i) {
        const std::string k_name = cache_name("out_cache_", "k", i);
        const std::string v_name = cache_name("out_cache_", "v", i);
        VulkanDecoderKvLayer& next = next_cache[static_cast<std::size_t>(i)];
        if (ex.extract(k_name.c_str(), next.key, command) != 0 or next.key.empty() or
            ex.extract(v_name.c_str(), next.value, command) != 0 or next.value.empty()) [[unlikely]] {
            throw std::runtime_error("failed to extract decoder Vulkan kv cache");
        }
    }

    ncnn::VkMat out_gpu;
    if (ex.extract("out0", out_gpu, command) != 0 or out_gpu.empty()) [[unlikely]] {
        throw std::runtime_error("failed to extract decoder Vulkan out0");
    }

    ncnn::Option download_option = kv_cache.command_option(net.opt);
    download_option.use_packing_layout = false;
    ncnn::Mat out;
    command.record_download(out_gpu, out, download_option);
    if (command.submit_and_wait() != 0 or out.empty()) [[unlikely]] {
        throw std::runtime_error("failed to submit decoder Vulkan command");
    }
    profile.count("decoder_submit");
    kv_cache.replace(std::move(next_cache));
    return out;
}
#endif

} // namespace voxcpm2::runtime
