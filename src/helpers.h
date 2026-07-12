// Internal helper functions for VoxCPM2 runtime

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstring>
#include <initializer_list>
#include <print>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <mat.h>
#include <net.h>

namespace voxcpm2::runtime {

inline ncnn::Mat make_i64_input(int w, int h) {
    ncnn::Mat mat(w, h, std::size_t{8});
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

inline ncnn::Mat make_f32_input(std::span<const float> values) {
    ncnn::Mat mat(static_cast<int>(values.size()));
    std::memcpy(mat.data, values.data(), values.size() * sizeof(float));
    return mat;
}

inline ncnn::Mat make_f32_input(const std::vector<float>& values) {
    return make_f32_input(std::span<const float>(values));
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
    std::memcpy(mat.data, values.data(), values.size() * sizeof(float));
    return mat;
}

inline ncnn::Mat make_f32_mat(int w, int h, int c, const std::vector<float>& values) {
    if (static_cast<int>(values.size()) != w * h * c) [[unlikely]] {
        throw std::runtime_error("make_f32_mat size mismatch");
    }
    ncnn::Mat mat(w, h, c);
    const std::size_t plane = static_cast<std::size_t>(w) * h;
    for (int q = 0; q < c; ++q) {
        std::memcpy(mat.channel(q), values.data() + plane * q, plane * sizeof(float));
    }
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
    if (ex.input("in0", in) != 0) [[unlikely]] {
        throw std::runtime_error("failed to bind in0");
    }
    ncnn::Mat out;
    if (ex.extract("out0", out) != 0 or out.empty()) [[unlikely]] {
        throw std::runtime_error("failed to extract out0");
    }
    return out;
}

inline ncnn::Mat run_net(ncnn::Net& net, std::initializer_list<ncnn::Mat> inputs) {
    static constexpr std::array input_names = {"in0", "in1", "in2", "in3", "in4"};
    if (inputs.size() > input_names.size()) [[unlikely]] {
        throw std::runtime_error("too many component inputs");
    }
    ncnn::Extractor ex = net.create_extractor();
    std::size_t i = 0;
    for (const ncnn::Mat& input : inputs) {
        const char* name = input_names[i];
        if (ex.input(name, input) != 0) [[unlikely]] {
            throw std::runtime_error("failed to bind " + std::string(name));
        }
        ++i;
    }
    ncnn::Mat out;
    if (ex.extract("out0", out) != 0 or out.empty()) [[unlikely]] {
        throw std::runtime_error("failed to extract out0");
    }
    return out;
}

inline std::vector<float> run_net_vec(ncnn::Net& net, std::initializer_list<ncnn::Mat> inputs) {
    return mat_to_vector(run_net(net, inputs));
}

} // namespace voxcpm2::runtime
