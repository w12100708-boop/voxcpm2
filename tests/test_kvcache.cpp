// Decoder KV cache host/Vulkan regression tests

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "kvcache.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include <gpu.h>

namespace {

constexpr int kHeadDim = 4;
constexpr int kLayers = 2;

const char kDecoderParam[] = R"NCNN(7767517
7 18
Input                    in0                      0 1 in0
Input                    in1                      0 1 in1
Input                    kv_cache                 0 4 cache_k0 cache_v0 cache_k1 cache_v1
Split                    split0                   1 3 in0 q0 k0 v0
SDPA                     sdpa0                    6 3 q0 k0 v0 in1 cache_k0 cache_v0 hidden0 out_cache_k0 out_cache_v0 5=1 6=0.5 7=1
Split                    split1                   1 3 hidden0 q1 k1 v1
SDPA                     sdpa1                    6 3 q1 k1 v1 in1 cache_k1 cache_v1 out0 out_cache_k1 out_cache_v1 5=1 6=0.5 7=1
)NCNN";

void check(bool condition, const char* message) {
    if (not condition) {
        throw std::runtime_error(message);
    }
}

ncnn::Mat make_embed(int tokens, float offset) {
    ncnn::Mat embed(kHeadDim, tokens, 1);
    auto* values = static_cast<float*>(embed.data);
    for (int i = 0; i < kHeadDim * tokens; ++i) {
        values[i] = offset + static_cast<float>(i + 1) * 0.125f;
    }
    return embed;
}

std::vector<float> values(const ncnn::Mat& mat) {
    check(mat.elemsize == 4u and mat.elempack == 1, "decoder output must be unpacked fp32");
    const auto* begin = static_cast<const float*>(mat.data);
    return std::vector<float>(begin, begin + mat.total());
}

void assert_close(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance) {
    check(actual.size() == expected.size(), "decoder output size mismatch");
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(actual[i] - expected[i]));
    }
    if (max_diff > tolerance) {
        throw std::runtime_error("decoder host/Vulkan output mismatch");
    }
}

void configure_net(ncnn::Net& net, bool vulkan) {
    net.opt.num_threads = 1;
    net.opt.use_vulkan_compute = vulkan;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_storage = vulkan;
    net.opt.use_fp16_packed = vulkan;
    net.opt.use_fp16_arithmetic = vulkan;
    if (vulkan) {
        net.set_vulkan_device(0);
    }
    check(net.load_param_mem(kDecoderParam) == 0, "failed to load decoder test param");
    static const unsigned char empty_model[1] = {0};
    check(net.load_model(empty_model) == 0, "failed to load decoder test model");
}

std::vector<std::vector<float>> run_host() {
    ncnn::Net net;
    configure_net(net, false);
    voxcpm2::runtime::HostDecoderKvCache cache(kLayers, kHeadDim, 1);
    voxcpm2::runtime::Profile profile(false);
    std::vector<std::vector<float>> outputs;

    outputs.push_back(values(voxcpm2::runtime::run_decoder_with_kv(
        net,
        make_embed(2, 0.0f),
        voxcpm2::runtime::make_decoder_mask(2, 0),
        nullptr,
        nullptr,
        cache,
        kLayers,
        profile)));
    check(cache.token_count() == 2, "host prefill token count mismatch");

    for (int step = 0; step < 2; ++step) {
        outputs.push_back(values(voxcpm2::runtime::run_decoder_with_kv(
            net,
            make_embed(1, 1.0f + static_cast<float>(step)),
            voxcpm2::runtime::make_decoder_mask(1, cache.token_count()),
            nullptr,
            nullptr,
            cache,
            kLayers,
            profile)));
        check(cache.token_count() == 3 + step, "host decode token count mismatch");
    }
    return outputs;
}

#if NCNN_VULKAN
void run_vulkan(const std::vector<std::vector<float>>& expected) {
    if (ncnn::get_gpu_count() == 0) {
        std::puts("test_kvcache: Vulkan unavailable, device checks skipped");
        return;
    }

    ncnn::create_gpu_instance();
    {
        ncnn::Net net;
        configure_net(net, true);
        voxcpm2::runtime::VulkanDecoderKvCache cache(net.vulkan_device(), kLayers, kHeadDim, 1);
        voxcpm2::runtime::Profile profile(false);

        assert_close(
            values(voxcpm2::runtime::run_decoder_with_kv(
                net,
                make_embed(2, 0.0f),
                voxcpm2::runtime::make_decoder_mask(2, 0),
                nullptr,
                nullptr,
                cache,
                kLayers,
                profile)),
            expected[0],
            2e-3f);
        check(cache.token_count() == 2, "Vulkan prefill token count mismatch");

        for (int step = 0; step < 2; ++step) {
            assert_close(
                values(voxcpm2::runtime::run_decoder_with_kv(
                    net,
                    make_embed(1, 1.0f + static_cast<float>(step)),
                    voxcpm2::runtime::make_decoder_mask(1, cache.token_count()),
                    nullptr,
                    nullptr,
                    cache,
                    kLayers,
                    profile)),
                expected[static_cast<std::size_t>(step + 1)],
                2e-3f);
            check(cache.token_count() == 3 + step, "Vulkan decode token count mismatch");
        }

        const int stable_tokens = cache.token_count();
        std::vector<voxcpm2::runtime::VulkanDecoderKvLayer> invalid;
        invalid.reserve(kLayers);
        for (int i = 0; i < kLayers; ++i) {
            invalid.push_back(cache.layer(i));
        }
        invalid.front().key.w = kHeadDim - 1;
        bool rejected = false;
        try {
            cache.replace(std::move(invalid));
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected, "invalid Vulkan cache replacement was accepted");
        check(cache.token_count() == stable_tokens, "failed Vulkan replacement mutated cache");
    }
    ncnn::destroy_gpu_instance();
}
#endif

} // namespace

int main() {
    try {
        const std::vector<std::vector<float>> expected = run_host();
#if NCNN_VULKAN
        run_vulkan(expected);
#endif
        std::puts("test_kvcache: ok");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "test_kvcache: %s\n", error.what());
        return 1;
    }
}
