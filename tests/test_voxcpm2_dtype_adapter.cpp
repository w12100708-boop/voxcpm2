// VoxCPM2 dtype adapter custom layer regression tests

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "ncnn_layers/dtype_adapter/voxcpm2_dtype_adapter.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include <mat.h>
#include <net.h>

namespace {

ncnn::Mat make_f32() {
    ncnn::Mat mat(4, 3);
    float* ptr = static_cast<float*>(mat.data);
    for (int i = 0; i < 12; ++i) {
        ptr[i] = static_cast<float>(i) * 0.25f - 1.0f;
    }
    return mat;
}

ncnn::Mat make_f16() {
    ncnn::Mat mat(4, 3, static_cast<std::size_t>(2u));
    unsigned short* ptr = static_cast<unsigned short*>(mat.data);
    for (int i = 0; i < 12; ++i) {
        ptr[i] = ncnn::float32_to_float16(static_cast<float>(i) * 0.5f - 2.0f);
    }
    return mat;
}

void check(bool cond, const char* msg) {
    if (not cond) {
        throw std::runtime_error(msg);
    }
}

void check_split_rmsnorm_graph_keeps_bypass() {
    const std::string param =
        "7767517\n"
        "5 6\n"
        "Input                    in0                      0 1 in0\n"
        "Split                    split0                   1 2 in0 bypass to_norm\n"
        "VoxCPM2DTypeAdapter      adapter0                 1 1 to_norm fp32_norm\n"
        "RMSNorm                  rms0                     1 1 fp32_norm normed 0=4 1=1.000000e-5 2=0\n"
        "Noop                     out0                     1 1 bypass out0\n";

    ncnn::Net net;
    voxcpm2::runtime::register_voxcpm2_dtype_adapter(net);
    check(net.load_param_mem(param.c_str()) == 0, "split graph load_param failed");
    const unsigned char empty_bin[] = {0};
    check(net.load_model(empty_bin) == 0, "split graph load_model failed");

    ncnn::Mat input = make_f32();
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(false);
    ex.input("in0", input);

    ncnn::Mat bypass;
    ncnn::Mat normed;
    check(ex.extract("out0", bypass) == 0 and not bypass.empty(), "extract bypass failed");
    check(ex.extract("normed", normed) == 0 and not normed.empty(), "extract normed failed");

    check(bypass.elemsize == 4u, "bypass elemsize mismatch");
    check(normed.elemsize == 4u, "normed elemsize mismatch");
    const float* input_ptr = static_cast<const float*>(input.data);
    const float* bypass_ptr = static_cast<const float*>(bypass.data);
    const float* normed_ptr = static_cast<const float*>(normed.data);
    bool norm_changed = false;
    for (int i = 0; i < 12; ++i) {
        check(std::abs(bypass_ptr[i] - input_ptr[i]) < 1e-7f, "bypass was modified by RMSNorm path");
        norm_changed = norm_changed or std::abs(normed_ptr[i] - input_ptr[i]) > 1e-4f;
    }
    check(norm_changed, "RMSNorm path did not change values");
}

} // namespace

int main() {
    try {
        voxcpm2::runtime::VoxCPM2DTypeAdapter layer;
        ncnn::Option opt;

        ncnn::Mat in_f32 = make_f32();
        ncnn::Mat out_f32;
        check(layer.forward(in_f32, out_f32, opt) == 0, "fp32 forward failed");
        check(out_f32.elemsize == 4u, "fp32 output elemsize mismatch");
        check(out_f32.w == in_f32.w and out_f32.h == in_f32.h and out_f32.dims == in_f32.dims, "fp32 output shape mismatch");
        check(out_f32.data != in_f32.data, "fp32 output aliases input");

        float* out_ptr = static_cast<float*>(out_f32.data);
        const float* in_ptr = static_cast<const float*>(in_f32.data);
        for (int i = 0; i < 12; ++i) {
            check(std::abs(out_ptr[i] - in_ptr[i]) < 1e-7f, "fp32 output value mismatch");
        }
        out_ptr[0] = 123.0f;
        check(std::abs(in_ptr[0] - (-1.0f)) < 1e-7f, "fp32 output mutation changed input");

        ncnn::Mat in_f16 = make_f16();
        ncnn::Mat out_f16;
        opt.use_bf16_storage = false;
        check(layer.forward(in_f16, out_f16, opt) == 0, "fp16 forward failed");
        check(out_f16.elemsize == 4u, "fp16 output elemsize mismatch");
        const float* f16_ptr = static_cast<const float*>(out_f16.data);
        for (int i = 0; i < 12; ++i) {
            const float expected = static_cast<float>(i) * 0.5f - 2.0f;
            check(std::abs(f16_ptr[i] - expected) < 1e-3f, "fp16 output value mismatch");
        }

        check_split_rmsnorm_graph_keeps_bypass();
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
