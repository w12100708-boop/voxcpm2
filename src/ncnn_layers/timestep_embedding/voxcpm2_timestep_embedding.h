// Local VoxCPM2 timestep embedding custom layer for ncnn.
//
// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#ifndef VOXCPM2_LAYER_TIMESTEP_EMBEDDING_H
#define VOXCPM2_LAYER_TIMESTEP_EMBEDDING_H

#include "gpu.h"
#include "layer.h"
#include "pipeline.h"

#include <net.h>

namespace voxcpm2::runtime {

class VoxCPM2TimestepEmbedding : public ncnn::Layer
{
public:
    VoxCPM2TimestepEmbedding();

    virtual int load_param(const ncnn::ParamDict& pd);
    virtual int create_pipeline(const ncnn::Option& opt);
    virtual int destroy_pipeline(const ncnn::Option& opt);

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const;
#if NCNN_VULKAN
    virtual int forward(const ncnn::VkMat& bottom_blob, ncnn::VkMat& top_blob, ncnn::VkCompute& cmd, const ncnn::Option& opt) const;
#endif

public:
    int half_dim;
    float scale;
    float max_period;
    int schedule_mode;

#if NCNN_VULKAN
    ncnn::Pipeline* pipeline_embedding;
#endif
};

int register_voxcpm2_timestep_embedding(ncnn::Net& net);

} // namespace voxcpm2::runtime

#endif // VOXCPM2_LAYER_TIMESTEP_EMBEDDING_H
