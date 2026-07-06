// Copyright 2025 Tencent
// Modified for voxcpm2-ncnn.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef VOXCPM2_LAYER_SDPA_H
#define VOXCPM2_LAYER_SDPA_H

#include "gpu.h"
#include "layer.h"
#include "pipeline.h"

#include <net.h>

namespace voxcpm2::runtime {

class VoxCPM2SDPA : public ncnn::Layer
{
public:
    VoxCPM2SDPA();

    virtual int load_param(const ncnn::ParamDict& pd);
    virtual int create_pipeline(const ncnn::Option& opt);
    virtual int destroy_pipeline(const ncnn::Option& opt);

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const;
#if NCNN_VULKAN
    virtual int forward(const std::vector<ncnn::VkMat>& bottom_blobs, std::vector<ncnn::VkMat>& top_blobs, ncnn::VkCompute& cmd, const ncnn::Option& opt) const;
#endif

public:
    int attn_mask;
    float scale;
    int kv_cache;

    int int8_scale_term;

#if NCNN_VULKAN
    ncnn::Pipeline* pipeline_sdpa_qk_cross;
    ncnn::Pipeline* pipeline_sdpa_qkv_cross;
    ncnn::Pipeline* pipeline_sdpa_fa[8];
    ncnn::Layer* qk_softmax;

    bool use_flash_attention;
    int FA_coopmat_M;
    int FA_coopmat_N;
    int FA_coopmat_K;
    int FA_coopmat_subgroup_size;
    int FA_UNROLL_SG_M;
    int FA_UNROLL_WG_M;

    bool use_cooperative_matrix;
    int coopmat_M;
    int coopmat_N;
    int coopmat_K;
    int coopmat_subgroup_size;
    int UNROLL_SG_M;
    int UNROLL_SG_N;
    int UNROLL_SG_K;
    int UNROLL_WG_M;
    int UNROLL_WG_N;
#endif
};

int register_voxcpm2_sdpa(ncnn::Net& net);
void set_voxcpm2_sdpa_flash_attention_enabled(bool enabled);

} // namespace voxcpm2::runtime

#endif // VOXCPM2_LAYER_SDPA_H
