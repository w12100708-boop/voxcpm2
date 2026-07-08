// VoxCPM2 SDPA custom layer regression tests

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "helpers.h"
#include "ncnn_layers/sdpa/voxcpm2_sdpa.h"

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

namespace {

std::vector<float> values(int count, float offset) {
    std::vector<float> out(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        out[static_cast<std::size_t>(i)] = std::sin(static_cast<float>(i) * 0.17f + offset);
    }
    return out;
}

std::string sdpa_param(bool native) {
    return std::string("7767517\n")
        + "4 4\n"
        + "Input                    in0                      0 1 in0\n"
        + "Input                    in1                      0 1 in1\n"
        + "Input                    in2                      0 1 in2\n"
        + (native
               ? "SDPA                     sdpa                     3 1 in0 in1 in2 out0 5=0\n"
               : "VoxCPM2SDPA              sdpa                     3 1 in0 in1 in2 out0 5=0\n");
}

ncnn::Mat run_graph(bool native, bool vulkan) {
    ncnn::Net net;
    net.opt.use_vulkan_compute = vulkan;
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
    voxcpm2::runtime::register_voxcpm2_sdpa(net);

    const std::string param = sdpa_param(native);
    if (net.load_param_mem(param.c_str()) != 0) {
        throw std::runtime_error("failed to load sdpa test param");
    }
    const unsigned char empty_bin[] = {0};
    net.load_model(empty_bin);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", voxcpm2::runtime::make_f32_mat(4, 2, 2, values(16, 0.1f)));
    ex.input("in1", voxcpm2::runtime::make_f32_mat(4, 2, 1, values(8, 0.2f)));
    ex.input("in2", voxcpm2::runtime::make_f32_mat(4, 2, 1, values(8, 0.3f)));

    ncnn::Mat out;
    if (ex.extract("out0", out) != 0 or out.empty()) {
        throw std::runtime_error("failed to extract sdpa output");
    }
    return out;
}

std::vector<float> reference() {
    const std::vector<float> query = values(16, 0.1f);
    const std::vector<float> key = values(8, 0.2f);
    const std::vector<float> value = values(8, 0.3f);
    std::vector<float> out(16);
    const float scale = 1.0f / std::sqrt(4.0f);

    for (int head = 0; head < 2; ++head) {
        for (int i = 0; i < 2; ++i) {
            float scores[2] = {};
            for (int j = 0; j < 2; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    sum += query[static_cast<std::size_t>(head) * 8 + i * 4 + k] * key[static_cast<std::size_t>(j) * 4 + k];
                }
                scores[j] = sum * scale;
            }
            const float max_score = std::max(scores[0], scores[1]);
            scores[0] = std::exp(scores[0] - max_score);
            scores[1] = std::exp(scores[1] - max_score);
            const float denom = scores[0] + scores[1];
            scores[0] /= denom;
            scores[1] /= denom;

            for (int d = 0; d < 4; ++d) {
                out[static_cast<std::size_t>(head) * 8 + i * 4 + d] =
                    scores[0] * value[static_cast<std::size_t>(d)] +
                    scores[1] * value[static_cast<std::size_t>(4 + d)];
            }
        }
    }
    return out;
}

void assert_close(const std::vector<float>& actual, const std::vector<float>& expected, float limit, const char* label) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(std::string(label) + " size mismatch: actual=" + std::to_string(actual.size()) + " expected=" + std::to_string(expected.size()));
    }
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(actual[i] - expected[i]));
    }
    if (max_diff > limit) {
        throw std::runtime_error(std::string(label) + " diff too large: " + std::to_string(max_diff));
    }
}

} // namespace

int main() {
    try {
        const std::vector<float> ref = reference();
        const std::vector<float> custom_cpu = voxcpm2::runtime::mat_to_vector(run_graph(false, false));
        const std::vector<float> native_cpu = voxcpm2::runtime::mat_to_vector(run_graph(true, false));
        assert_close(custom_cpu, ref, 1e-5f, "custom cpu");
        assert_close(native_cpu, ref, 1e-5f, "native cpu");

#if NCNN_VULKAN
        ncnn::create_gpu_instance();
        if (ncnn::get_gpu_count() > 0) {
            const std::vector<float> custom_vk = voxcpm2::runtime::mat_to_vector(run_graph(false, true));
            const std::vector<float> native_vk = voxcpm2::runtime::mat_to_vector(run_graph(true, true));
            assert_close(custom_vk, native_vk, 2e-3f, "custom/native vulkan");
        }
        ncnn::destroy_gpu_instance();
#endif

        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
#if NCNN_VULKAN
        ncnn::destroy_gpu_instance();
#endif
        return 1;
    }
}
