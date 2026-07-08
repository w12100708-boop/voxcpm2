// Internal helper functions for VoxCPM2 runtime

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstring>
#include <format>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <mat.h>
#include <net.h>

namespace voxcpm2::runtime {

// Components with separate .cpu/.vulkan param variants that run fp16 on Vulkan
// via VoxCPM2DTypeAdapter-wrapped RMSNorm, and stay fp32 on CPU.
inline constexpr std::array<const char*, 4> kDualBackendComponents = {
    "dit_estimator", "base_decoder_kv", "residual_decoder_kv", "feat_encoder",
};

constexpr bool is_dual_backend_component(std::string_view name) {
    for (auto* c : kDualBackendComponents) {
        if (name == c) {
            return true;
        }
    }
    return false;
}

inline std::string resolve_param_key(const std::string& name, bool use_vulkan) {
    if (is_dual_backend_component(name)) {
        return use_vulkan ? name + ".vulkan" : name + ".cpu";
    }
    return name;
}

inline ncnn::Mat make_i64_input(int w, int h) {
    ncnn::Mat mat(w, h, std::size_t{8});
    std::memset(mat.data, 0, mat.total() * mat.elemsize);
    auto* p = static_cast<long long*>(mat.data);
    for (int i = 0; i < w * h; ++i) {
        p[i] = (i % 2 == 0) ? 1 : 0;
    }
    return mat;
}

inline ncnn::Mat make_i32_input(const std::vector<int>& values) {
    ncnn::Mat mat(static_cast<int>(values.size()));
    std::memcpy(mat.data, values.data(), values.size() * sizeof(int));
    return mat;
}

inline ncnn::Mat make_f32_input(const std::vector<float>& values) {
    ncnn::Mat mat(static_cast<int>(values.size()));
    std::memset(mat.data, 0, mat.total() * mat.elemsize);
    std::memcpy(mat.data, values.data(), values.size() * sizeof(float));
    return mat;
}

inline ncnn::Mat make_f32_input(int w, int h, int c = 1) {
    ncnn::Mat mat = c == 1 ? ncnn::Mat(w, h) : ncnn::Mat(w, h, c);
    auto* p = static_cast<float*>(mat.data);
    for (int i = 0; i < mat.total(); ++i) {
        p[i] = static_cast<float>((i % 17) - 8) / 32.0f;
    }
    return mat;
}

inline ncnn::Mat make_f32_mat(int w, int h, const std::vector<float>& values) {
    if (static_cast<int>(values.size()) != w * h) [[unlikely]] {
        throw std::runtime_error("make_f32_mat size mismatch");
    }
    ncnn::Mat mat(w, h);
    std::memset(mat.data, 0, mat.total() * mat.elemsize);
    std::memcpy(mat.data, values.data(), values.size() * sizeof(float));
    return mat;
}

inline ncnn::Mat make_f32_mat(int w, int h, int c, const std::vector<float>& values) {
    if (static_cast<int>(values.size()) != w * h * c) [[unlikely]] {
        throw std::runtime_error("make_f32_mat size mismatch");
    }
    ncnn::Mat mat(w, h, c);
    std::memset(mat.data, 0, mat.total() * mat.elemsize);
    std::memcpy(mat.data, values.data(), values.size() * sizeof(float));
    return mat;
}

inline std::vector<float> mat_to_vector(const ncnn::Mat& mat) {
    if (mat.elemsize != 4) [[unlikely]] {
        throw std::runtime_error("expected f32 ncnn::Mat");
    }
    if (mat.dims == 1) {
        std::vector<float> values(static_cast<std::size_t>(mat.w));
        std::memcpy(values.data(), mat.data, values.size() * sizeof(float));
        return values;
    }
    if (mat.dims == 2) {
        std::vector<float> values(static_cast<std::size_t>(mat.w) * mat.h);
        std::memcpy(values.data(), mat.data, values.size() * sizeof(float));
        return values;
    }
    if (mat.dims == 3) {
        const std::size_t plane = static_cast<std::size_t>(mat.w) * mat.h;
        std::vector<float> values(plane * mat.c);
        for (int q = 0; q < mat.c; ++q) {
            const float* src = mat.channel(q);
            std::memcpy(values.data() + plane * q, src, plane * sizeof(float));
        }
        return values;
    }
    std::vector<float> values(mat.total());
    std::memcpy(values.data(), mat.data, values.size() * sizeof(float));
    return values;
}

inline void print_mat_shape(const std::string& name, const ncnn::Mat& mat) {
    if (mat.empty()) {
        std::println(stderr, "{}: empty", name);
        return;
    }
    std::println(
        stderr,
        "{}: dims={} w={} h={} c={} total={} elemsize={}",
        name,
        mat.dims,
        mat.w,
        mat.h,
        mat.c,
        mat.total(),
        mat.elemsize);
}

inline ncnn::Mat run_single_input(ncnn::Net& net, const ncnn::Mat& in) {
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;
    if (ex.extract("out0", out) != 0 or out.empty()) [[unlikely]] {
        throw std::runtime_error("failed to extract out0");
    }
    return out;
}

inline ncnn::Mat run_net(ncnn::Net& net, const std::vector<ncnn::Mat>& inputs) {
    ncnn::Extractor ex = net.create_extractor();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const std::string name = std::format("in{}", i);
        ex.input(name.c_str(), inputs[i]);
    }
    ncnn::Mat out;
    if (ex.extract("out0", out) != 0 or out.empty()) [[unlikely]] {
        throw std::runtime_error("failed to extract out0");
    }
    return out;
}

inline std::vector<float> run_net_vec(ncnn::Net& net, const std::vector<ncnn::Mat>& inputs) {
    return mat_to_vector(run_net(net, inputs));
}

} // namespace voxcpm2::runtime
