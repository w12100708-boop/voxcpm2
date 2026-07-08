// Local dtype adapter layer for VoxCPM2 ncnn graphs.
//
// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "voxcpm2_dtype_adapter.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include <mat.h>

namespace voxcpm2::runtime {
namespace {

#if NCNN_VULKAN
constexpr char kDTypeAdapterComp[] = {
#embed "voxcpm2_dtype_adapter.glsl"
    , 0
};
#endif

ncnn::Layer* create_voxcpm2_dtype_adapter(void*) {
    return new VoxCPM2DTypeAdapter;
}

std::size_t scalar_elemsize(const ncnn::Mat& mat) {
    return mat.elemsize / static_cast<std::size_t>(std::max(mat.elempack, 1));
}

void create_like_fp32(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) {
    const std::size_t elemsize = static_cast<std::size_t>(4u) * std::max(bottom_blob.elempack, 1);
    if (bottom_blob.dims == 1) {
        top_blob.create(bottom_blob.w, elemsize, bottom_blob.elempack, opt.blob_allocator);
    } else if (bottom_blob.dims == 2) {
        top_blob.create(bottom_blob.w, bottom_blob.h, elemsize, bottom_blob.elempack, opt.blob_allocator);
    } else if (bottom_blob.dims == 3) {
        top_blob.create(bottom_blob.w, bottom_blob.h, bottom_blob.c, elemsize, bottom_blob.elempack, opt.blob_allocator);
    } else if (bottom_blob.dims == 4) {
        top_blob.create(bottom_blob.w, bottom_blob.h, bottom_blob.d, bottom_blob.c, elemsize, bottom_blob.elempack, opt.blob_allocator);
    }
}

#if NCNN_VULKAN
std::size_t scalar_elemsize(const ncnn::VkMat& mat) {
    return mat.elemsize / static_cast<std::size_t>(std::max(mat.elempack, 1));
}

void create_like_fp32(const ncnn::VkMat& bottom_blob, ncnn::VkMat& top_blob, const ncnn::Option& opt) {
    const std::size_t elemsize = static_cast<std::size_t>(4u) * std::max(bottom_blob.elempack, 1);
    if (bottom_blob.dims == 1) {
        top_blob.create(bottom_blob.w, elemsize, bottom_blob.elempack, opt.blob_vkallocator);
    } else if (bottom_blob.dims == 2) {
        top_blob.create(bottom_blob.w, bottom_blob.h, elemsize, bottom_blob.elempack, opt.blob_vkallocator);
    } else if (bottom_blob.dims == 3) {
        top_blob.create(bottom_blob.w, bottom_blob.h, bottom_blob.c, elemsize, bottom_blob.elempack, opt.blob_vkallocator);
    } else if (bottom_blob.dims == 4) {
        top_blob.create(bottom_blob.w, bottom_blob.h, bottom_blob.d, bottom_blob.c, elemsize, bottom_blob.elempack, opt.blob_vkallocator);
    }
}
#endif

} // namespace

VoxCPM2DTypeAdapter::VoxCPM2DTypeAdapter() {
    one_blob_only = true;
    support_inplace = false;
    support_packing = false;
#if NCNN_VULKAN
    support_vulkan = true;
    support_vulkan_packing = false;
    support_vulkan_any_packing = false;
    pipeline_fp16_to_fp32 = nullptr;
#endif
}

int VoxCPM2DTypeAdapter::create_pipeline(const ncnn::Option& opt) {
#if NCNN_VULKAN
    if (not opt.use_vulkan_compute or vkdev == nullptr) {
        return 0;
    }

    std::vector<unsigned int> spirv;
    int ret = ncnn::compile_spirv_module(kDTypeAdapterComp, static_cast<int>(sizeof(kDTypeAdapterComp) - 1), opt, spirv);
    if (ret != 0) {
        return ret;
    }

    std::vector<ncnn::vk_specialization_type> specializations(1);
    specializations[0].u32 = 0;

    pipeline_fp16_to_fp32 = new ncnn::Pipeline(vkdev);
    pipeline_fp16_to_fp32->set_local_size_xyz(64, 1, 1);
    return pipeline_fp16_to_fp32->create(spirv.data(), spirv.size() * sizeof(unsigned int), specializations);
#else
    (void)opt;
    return 0;
#endif
}

int VoxCPM2DTypeAdapter::destroy_pipeline(const ncnn::Option& opt) {
#if NCNN_VULKAN
    (void)opt;
    delete pipeline_fp16_to_fp32;
    pipeline_fp16_to_fp32 = nullptr;
#else
    (void)opt;
#endif
    return 0;
}

int VoxCPM2DTypeAdapter::forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const {
    const std::size_t elemsize = scalar_elemsize(bottom_blob);
    if (elemsize != 4u and elemsize != 2u) {
        return -1;
    }

    create_like_fp32(bottom_blob, top_blob, opt);
    if (top_blob.empty()) {
        return -100;
    }

    float* dst = static_cast<float*>(top_blob.data);
    const std::size_t total = bottom_blob.total();
    if (elemsize == 4u) {
        const float* src = static_cast<const float*>(bottom_blob.data);
        std::memcpy(dst, src, total * sizeof(float));
        return 0;
    }

    const unsigned short* src = static_cast<const unsigned short*>(bottom_blob.data);
    for (std::size_t i = 0; i < total; ++i) {
        dst[i] = opt.use_bf16_storage ? ncnn::bfloat16_to_float32(src[i]) : ncnn::float16_to_float32(src[i]);
    }
    return 0;
}

#if NCNN_VULKAN
int VoxCPM2DTypeAdapter::forward(const ncnn::VkMat& bottom_blob,
                               ncnn::VkMat& top_blob,
                               ncnn::VkCompute& cmd,
                               const ncnn::Option& opt) const {
    const std::size_t elemsize = scalar_elemsize(bottom_blob);
    if (elemsize == 4u) {
        cmd.record_clone(bottom_blob, top_blob, opt);
        return 0;
    }
    if (elemsize != 2u) {
        return -1;
    }

    create_like_fp32(bottom_blob, top_blob, opt);
    if (top_blob.empty()) {
        return -100;
    }

    std::vector<ncnn::VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<ncnn::vk_constant_type> constants(1);
    constants[0].u32 = static_cast<unsigned int>(bottom_blob.total());

    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(bottom_blob.total());
    dispatcher.h = 1;
    dispatcher.c = 1;
    cmd.record_pipeline(pipeline_fp16_to_fp32, bindings, constants, dispatcher);
    return 0;
}
#endif

int register_voxcpm2_dtype_adapter(ncnn::Net& net) {
    return net.register_custom_layer("VoxCPM2DTypeAdapter", create_voxcpm2_dtype_adapter);
}

} // namespace voxcpm2::runtime
