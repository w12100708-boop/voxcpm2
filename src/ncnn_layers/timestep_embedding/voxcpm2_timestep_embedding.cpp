// Local VoxCPM2 timestep embedding custom layer for ncnn.
//
// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "voxcpm2_timestep_embedding.h"

#include "../spirv_cache.h"
#include "../../embed_support.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <mat.h>
#include <modelbin.h>
#include <paramdict.h>

namespace voxcpm2::runtime {
namespace {

constexpr float kPi = 3.14159265358979323846f;

#if NCNN_VULKAN
constexpr char kTimestepEmbeddingComp[] = {
#embed "voxcpm2_timestep_embedding.glsl"
    , 0
};
#endif

ncnn::Layer* create_voxcpm2_timestep_embedding(void*) {
    return new VoxCPM2TimestepEmbedding;
}

std::size_t scalar_elemsize(const ncnn::Mat& mat) {
    const int elempack = std::max(mat.elempack, 1);
    return mat.elemsize / static_cast<std::size_t>(elempack);
}

float read_scalar(const ncnn::Mat& mat, std::size_t index, const ncnn::Option& opt) {
    const std::size_t elemsize = scalar_elemsize(mat);
    if (elemsize == 4u) {
        const float* ptr = static_cast<const float*>(mat.data);
        return ptr[index];
    }
    if (elemsize == 2u) {
        const unsigned short* ptr = static_cast<const unsigned short*>(mat.data);
        return opt.use_bf16_storage ? ncnn::bfloat16_to_float32(ptr[index]) : ncnn::float16_to_float32(ptr[index]);
    }
    return 0.0f;
}

} // namespace

VoxCPM2TimestepEmbedding::VoxCPM2TimestepEmbedding() {
    one_blob_only = true;
    support_inplace = false;
    support_vulkan = true;
    support_packing = false;
#if NCNN_VULKAN
    support_vulkan = true;
    support_vulkan_packing = false;
    support_vulkan_any_packing = false;
    pipeline_embedding = nullptr;
#endif

    half_dim = 512;
    scale = 1000.0f;
    max_period = 10000.0f;
    schedule_mode = 0;
}

int VoxCPM2TimestepEmbedding::load_param(const ncnn::ParamDict& pd) {
    half_dim = pd.get(0, 512);
    scale = pd.get(1, 1000.0f);
    max_period = pd.get(2, 10000.0f);
    schedule_mode = pd.get(3, 0);
    return 0;
}

int VoxCPM2TimestepEmbedding::create_pipeline(const ncnn::Option& opt) {
#if NCNN_VULKAN
    if (not opt.use_vulkan_compute or vkdev == nullptr) {
        return 0;
    }

    SpirvModule spirv;
    int ret = get_cached_spirv(
        SpirvShader::timestep_embedding,
        kTimestepEmbeddingComp,
        static_cast<int>(sizeof(kTimestepEmbeddingComp) - 1),
        opt,
        spirv);
    if (ret != 0) {
        return ret;
    }

    std::vector<ncnn::vk_specialization_type> specializations(4);
    specializations[0].i = half_dim;
    specializations[1].f = scale;
    specializations[2].f = std::log(max_period);
    specializations[3].i = schedule_mode;

    pipeline_embedding = new ncnn::Pipeline(vkdev);
    pipeline_embedding->set_local_size_xyz(64, 1, 1);
    return pipeline_embedding->create(
        spirv->data(),
        spirv->size() * sizeof(std::uint32_t),
        specializations);
#else
    (void)opt;
    return 0;
#endif
}

int VoxCPM2TimestepEmbedding::destroy_pipeline(const ncnn::Option& opt) {
#if NCNN_VULKAN
    (void)opt;
    delete pipeline_embedding;
    pipeline_embedding = nullptr;
#else
    (void)opt;
#endif
    return 0;
}

int VoxCPM2TimestepEmbedding::forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const {
    const int batch = bottom_blob.dims == 1 ? bottom_blob.w * bottom_blob.elempack : static_cast<int>(bottom_blob.total());
    const int out_dim = half_dim * 2;
    const std::size_t input_elemsize = scalar_elemsize(bottom_blob);
    if (input_elemsize != 4u and input_elemsize != 2u) {
        return -1;
    }
    top_blob.create(out_dim, batch, 4u, opt.blob_allocator);
    if (top_blob.empty()) {
        return -100;
    }

    float* top = static_cast<float*>(top_blob.data);
    const float log_period = std::log(max_period);
    float scheduled_t = 0.0f;
    if (schedule_mode != 0) {
        const float step_index = read_scalar(bottom_blob, 0, opt);
        const float timesteps = std::max(read_scalar(bottom_blob, 1, opt), 1.0f);
        const float u = 1.0f - step_index / timesteps;
        scheduled_t = u + (std::cos(kPi * 0.5f * u) - 1.0f + u);
    }
    for (int b = 0; b < batch; ++b) {
        const float t = (schedule_mode != 0 ? scheduled_t : read_scalar(bottom_blob, static_cast<std::size_t>(b), opt)) * scale;
        float* row = top + static_cast<std::size_t>(b) * out_dim;
        for (int i = 0; i < half_dim; ++i) {
            const float inv_freq = std::exp(-log_period * static_cast<float>(i) / static_cast<float>(half_dim - 1));
            const float phase = t * inv_freq;
            row[i] = std::sin(phase);
            row[i + half_dim] = std::cos(phase);
        }
    }
    return 0;
}

#if NCNN_VULKAN
int VoxCPM2TimestepEmbedding::forward(const ncnn::VkMat& bottom_blob,
                                      ncnn::VkMat& top_blob,
                                      ncnn::VkCompute& cmd,
                                      const ncnn::Option& opt) const {
    const int batch = bottom_blob.dims == 1 ? bottom_blob.w * bottom_blob.elempack : bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c;
    const int out_dim = half_dim * 2;
    const std::size_t elemsize = opt.use_fp16_storage ? 2u : 4u;
    top_blob.create(out_dim, batch, elemsize, opt.blob_vkallocator);
    if (top_blob.empty()) {
        return -100;
    }

    std::vector<ncnn::VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<ncnn::vk_constant_type> constants;

    ncnn::VkMat dispatcher;
    dispatcher.w = out_dim;
    dispatcher.h = batch;
    dispatcher.c = 1;
    cmd.record_pipeline(pipeline_embedding, bindings, constants, dispatcher);
    return 0;
}
#endif

int register_voxcpm2_timestep_embedding(ncnn::Net& net) {
    return net.register_custom_layer("VoxCPM2TimestepEmbedding", create_voxcpm2_timestep_embedding);
}

} // namespace voxcpm2::runtime
