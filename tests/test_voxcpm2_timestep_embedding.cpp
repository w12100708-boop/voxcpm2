// VoxCPM2 timestep embedding custom layer regression tests

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "ncnn_layers/timestep_embedding/voxcpm2_timestep_embedding.h"
#include "ncnn_layers/spirv_cache.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gpu.h>
#include <mat.h>
#include <net.h>
#include <paramdict.h>

namespace {

std::string timestep_param(int schedule_mode) {
    return std::string("7767517\n")
        + "2 2\n"
        + "Input                    in0                      0 1 in0\n"
        + "VoxCPM2TimestepEmbedding time_emb                 1 1 in0 out0 0=512 1=1000.0 2=10000.0 3=" + std::to_string(schedule_mode) + "\n";
}

ncnn::Mat make_input(float first, float second) {
    ncnn::Mat in(2);
    float* ptr = static_cast<float*>(in.data);
    ptr[0] = first;
    ptr[1] = second;
    return in;
}

ncnn::Mat make_fp16_input(float first, float second) {
    ncnn::Mat in(2, static_cast<std::size_t>(2u));
    unsigned short* ptr = static_cast<unsigned short*>(in.data);
    ptr[0] = ncnn::float32_to_float16(first);
    ptr[1] = ncnn::float32_to_float16(second);
    return in;
}

ncnn::Mat run_layer_fp16_cpu(int schedule_mode, float first, float second) {
    voxcpm2::runtime::VoxCPM2TimestepEmbedding layer;
    ncnn::ParamDict pd;
    pd.set(0, 512);
    pd.set(1, 1000.0f);
    pd.set(2, 10000.0f);
    pd.set(3, schedule_mode);
    if (layer.load_param(pd) != 0) {
        throw std::runtime_error("failed to load timestep embedding layer params");
    }

    ncnn::Option opt;
    opt.use_fp16_storage = true;
    opt.use_bf16_storage = false;

    ncnn::Mat out;
    if (layer.forward(make_fp16_input(first, second), out, opt) != 0 or out.empty()) {
        throw std::runtime_error("failed to run fp16 cpu timestep embedding layer");
    }
    return out;
}

ncnn::Mat run_graph(bool vulkan, int schedule_mode, float first, float second) {
    ncnn::Net net;
    net.opt.use_vulkan_compute = vulkan;
    net.opt.use_fp16_storage = true;
    net.opt.use_fp16_packed = true;
    net.opt.use_fp16_arithmetic = true;
    net.opt.use_bf16_storage = false;
    net.opt.use_bf16_packed = false;
    net.opt.use_int8_inference = false;
    net.opt.use_int8_storage = false;
    net.opt.use_int8_packed = false;
    net.opt.use_int8_arithmetic = false;
#if NCNN_VULKAN
    if (vulkan) {
        net.set_vulkan_device(0);
    }
#else
    if (vulkan) {
        throw std::runtime_error("Vulkan unavailable");
    }
#endif
    voxcpm2::runtime::register_voxcpm2_timestep_embedding(net);

    const std::string param = timestep_param(schedule_mode);
    if (net.load_param_mem(param.c_str()) != 0) {
        throw std::runtime_error("failed to load timestep embedding test param");
    }
    const unsigned char empty_bin[] = {0};
    net.load_model(empty_bin);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", make_input(first, second));

    ncnn::Mat out;
    if (ex.extract("out0", out) != 0 or out.empty()) {
        throw std::runtime_error("failed to extract timestep embedding output");
    }
    return out;
}

float schedule_t(float step_index, float timesteps) {
    const float u = 1.0f - step_index / timesteps;
    return u + (std::cos(3.14159265358979323846f * 0.5f * u) - 1.0f + u);
}

std::vector<float> reference(const float timesteps[2]) {
    std::vector<float> out(2 * 1024);
    const float log_period = std::log(10000.0f);
    for (int b = 0; b < 2; ++b) {
        const float t = timesteps[b] * 1000.0f;
        for (int i = 0; i < 512; ++i) {
            const float inv_freq = std::exp(-log_period * static_cast<float>(i) / 511.0f);
            const float phase = t * inv_freq;
            out[static_cast<std::size_t>(b) * 1024 + i] = std::sin(phase);
            out[static_cast<std::size_t>(b) * 1024 + 512 + i] = std::cos(phase);
        }
    }
    return out;
}

void check_close(const ncnn::Mat& mat, const std::vector<float>& ref, float tol, const char* label) {
    if (mat.total() != ref.size()) {
        throw std::runtime_error(std::string(label) + ": output size mismatch total=" + std::to_string(mat.total()) + " expected=" + std::to_string(ref.size()));
    }
    const float* ptr = static_cast<const float*>(mat.data);
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (not std::isfinite(ptr[i])) {
            throw std::runtime_error(std::string(label) + ": non-finite output");
        }
        max_diff = std::max(max_diff, std::abs(ptr[i] - ref[i]));
    }
    if (max_diff > tol) {
        throw std::runtime_error(std::string(label) + ": max diff " + std::to_string(max_diff));
    }
}

} // namespace

int main() {
    try {
        const float raw_timesteps[] = {0.70710677f, 0.25f};
        const std::vector<float> raw_ref = reference(raw_timesteps);
        check_close(run_graph(false, 0, raw_timesteps[0], raw_timesteps[1]), raw_ref, 1e-6f, "cpu raw");

        const float scheduled = schedule_t(1.0f, 10.0f);
        const float scheduled_timesteps[] = {scheduled, scheduled};
        const std::vector<float> scheduled_ref = reference(scheduled_timesteps);
        check_close(run_graph(false, 1, 1.0f, 10.0f), scheduled_ref, 1e-6f, "cpu schedule");
        check_close(run_layer_fp16_cpu(0, raw_timesteps[0], raw_timesteps[1]), raw_ref, 8e-2f, "cpu fp16 raw");
        check_close(run_layer_fp16_cpu(1, 1.0f, 10.0f), scheduled_ref, 8e-2f, "cpu fp16 schedule");
#if NCNN_VULKAN
        ncnn::create_gpu_instance();
        try {
            const voxcpm2::runtime::SpirvCacheStats cache_before = voxcpm2::runtime::spirv_cache_stats();
            check_close(run_graph(true, 0, raw_timesteps[0], raw_timesteps[1]), raw_ref, 8e-2f, "vulkan raw");
            check_close(run_graph(true, 1, 1.0f, 10.0f), scheduled_ref, 8e-2f, "vulkan schedule");
            const voxcpm2::runtime::SpirvCacheStats cache_after = voxcpm2::runtime::spirv_cache_stats();
            if (cache_after.compile_calls - cache_before.compile_calls != 1) {
                throw std::runtime_error("timestep shader should compile exactly once");
            }
            if (cache_after.cache_hits - cache_before.cache_hits != 1) {
                throw std::runtime_error("second timestep graph should reuse cached SPIR-V");
            }
        } catch (...) {
            ncnn::destroy_gpu_instance();
            throw;
        }
        ncnn::destroy_gpu_instance();
#endif
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
