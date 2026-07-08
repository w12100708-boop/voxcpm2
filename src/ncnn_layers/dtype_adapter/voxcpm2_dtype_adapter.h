// Local dtype adapter layer for VoxCPM2 ncnn graphs.
//
// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#ifndef VOXCPM2_LAYER_DTYPE_ADAPTER_H
#define VOXCPM2_LAYER_DTYPE_ADAPTER_H

#include "gpu.h"
#include "layer.h"
#include "pipeline.h"

#include <net.h>

namespace voxcpm2::runtime {

class VoxCPM2DTypeAdapter : public ncnn::Layer
{
public:
    VoxCPM2DTypeAdapter();

    virtual int create_pipeline(const ncnn::Option& opt);
    virtual int destroy_pipeline(const ncnn::Option& opt);

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const;
#if NCNN_VULKAN
    virtual int forward(const ncnn::VkMat& bottom_blob, ncnn::VkMat& top_blob, ncnn::VkCompute& cmd, const ncnn::Option& opt) const;
#endif

#if NCNN_VULKAN
    ncnn::Pipeline* pipeline_fp16_to_fp32;
#endif
};

int register_voxcpm2_dtype_adapter(ncnn::Net& net);

} // namespace voxcpm2::runtime

#endif // VOXCPM2_LAYER_DTYPE_ADAPTER_H
