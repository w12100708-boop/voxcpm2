// Process-local SPIR-V cache for VoxCPM2 custom ncnn layers.
//
// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#if NCNN_VULKAN

#include <array>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gpu.h>
#include <net.h>

namespace voxcpm2::runtime {

enum class SpirvShader : std::uint8_t {
    dtype_adapter,
    sdpa_cross,
    sdpa_cross_cooperative_matrix,
    sdpa_flash_attention,
    sdpa_flash_attention_cooperative_matrix,
    timestep_embedding,
};

using SpirvModule = std::shared_ptr<const std::vector<std::uint32_t>>;

struct SpirvCacheStats {
    std::size_t entries = 0;
    std::uint64_t compile_calls = 0;
    std::uint64_t cache_hits = 0;
};

namespace spirv_cache_detail {

struct DeviceSignature {
    int index = -1;
    std::uint32_t api_version = 0;
    std::uint32_t driver_version = 0;
    std::uint32_t driver_id = 0;
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    std::array<std::uint8_t, VK_UUID_SIZE> pipeline_cache_uuid{};

    bool operator==(const DeviceSignature&) const = default;
};

struct CacheKey {
    SpirvShader shader = SpirvShader::dtype_adapter;
    DeviceSignature device;
    std::uint32_t option_bits = 0;

    bool operator==(const CacheKey&) const = default;
};

inline std::size_t hash_combine(std::size_t seed, std::size_t value) {
    return seed ^ (value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2));
}

struct CacheKeyHash {
    std::size_t operator()(const CacheKey& key) const {
        std::size_t hash = static_cast<std::size_t>(key.shader);
        hash = hash_combine(hash, static_cast<std::size_t>(key.device.index));
        hash = hash_combine(hash, key.device.api_version);
        hash = hash_combine(hash, key.device.driver_version);
        hash = hash_combine(hash, key.device.driver_id);
        hash = hash_combine(hash, key.device.vendor_id);
        hash = hash_combine(hash, key.device.device_id);
        for (std::uint8_t byte : key.device.pipeline_cache_uuid) {
            hash = hash_combine(hash, byte);
        }
        return hash_combine(hash, key.option_bits);
    }
};

struct CacheEntry {
    bool compiling = true;
    int status = -1;
    SpirvModule module;
    std::condition_variable ready;
};

struct CacheState {
    std::mutex mutex;
    std::unordered_map<CacheKey, std::shared_ptr<CacheEntry>, CacheKeyHash> entries;
    std::uint64_t compile_calls = 0;
    std::uint64_t cache_hits = 0;
};

inline CacheState& cache_state() {
    static CacheState state;
    return state;
}

inline std::uint32_t option_bits(const ncnn::Option& opt) {
    std::uint32_t bits = 0;
    const auto set = [&](bool enabled, int bit) {
        if (enabled) {
            bits |= std::uint32_t{1} << bit;
        }
    };
    set(opt.use_bf16_packed, 0);
    set(opt.use_bf16_storage, 1);
    set(opt.use_cooperative_matrix, 2);
    set(opt.use_fp16_arithmetic, 3);
    set(opt.use_fp16_packed, 4);
    set(opt.use_fp16_storage, 5);
    set(opt.use_fp16_uniform, 6);
    set(opt.use_int16_packed, 7);
    set(opt.use_int16_storage, 8);
    set(opt.use_int8_arithmetic, 9);
    set(opt.use_int8_packed, 10);
    set(opt.use_int8_storage, 11);
    set(opt.use_int8_uniform, 12);
    set(opt.use_shader_local_memory, 13);
    set(opt.use_subgroup_ops, 14);
    return bits;
}

inline DeviceSignature device_signature(const ncnn::Option& opt) {
    int device_index = opt.vulkan_device_index;
    const int gpu_count = ncnn::get_gpu_count();
    if (device_index < 0 or device_index >= gpu_count) {
        device_index = ncnn::get_default_gpu_index();
    }
    if (device_index < 0 or device_index >= gpu_count) {
        return {};
    }

    const ncnn::GpuInfo& info = ncnn::get_gpu_info(device_index);
    DeviceSignature signature{
        .index = device_index,
        .api_version = info.api_version(),
        .driver_version = info.driver_version(),
        .driver_id = info.driver_id(),
        .vendor_id = info.vendor_id(),
        .device_id = info.device_id(),
    };
    std::memcpy(
        signature.pipeline_cache_uuid.data(),
        info.pipeline_cache_uuid(),
        signature.pipeline_cache_uuid.size());
    return signature;
}

} // namespace spirv_cache_detail

inline int get_cached_spirv(
    SpirvShader shader,
    const char* source,
    int source_size,
    const ncnn::Option& opt,
    SpirvModule& output) {
    using namespace spirv_cache_detail;

    const CacheKey key{
        .shader = shader,
        .device = device_signature(opt),
        .option_bits = option_bits(opt),
    };
    if (key.device.index < 0) {
        return -1;
    }

    CacheState& state = cache_state();
    std::shared_ptr<CacheEntry> entry;
    {
        std::unique_lock lock(state.mutex);
        const auto found = state.entries.find(key);
        if (found != state.entries.end()) {
            ++state.cache_hits;
            entry = found->second;
            entry->ready.wait(lock, [&] { return not entry->compiling; });
            output = entry->module;
            return entry->status;
        }

        entry = std::make_shared<CacheEntry>();
        state.entries.emplace(key, entry);
        ++state.compile_calls;
    }

    auto compiled = std::make_shared<std::vector<std::uint32_t>>();
    int status = -1;
    try {
        status = ncnn::compile_spirv_module(source, source_size, opt, *compiled);
    } catch (...) {
        {
            std::lock_guard lock(state.mutex);
            entry->status = -1;
            entry->compiling = false;
            state.entries.erase(key);
        }
        entry->ready.notify_all();
        throw;
    }

    {
        std::lock_guard lock(state.mutex);
        entry->status = status;
        if (status == 0) {
            entry->module = std::move(compiled);
            output = entry->module;
        } else {
            state.entries.erase(key);
        }
        entry->compiling = false;
    }
    entry->ready.notify_all();
    return status;
}

inline SpirvCacheStats spirv_cache_stats() {
    spirv_cache_detail::CacheState& state = spirv_cache_detail::cache_state();
    std::lock_guard lock(state.mutex);
    return SpirvCacheStats{
        .entries = state.entries.size(),
        .compile_calls = state.compile_calls,
        .cache_hits = state.cache_hits,
    };
}

} // namespace voxcpm2::runtime

#endif // NCNN_VULKAN
