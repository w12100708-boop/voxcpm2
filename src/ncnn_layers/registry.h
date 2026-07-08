// Centralized registration of VoxCPM2 custom ncnn layers per component.
//
// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <net.h>
#include <string>

#include "dtype_adapter/voxcpm2_dtype_adapter.h"
#include "../helpers.h"
#include "sdpa/voxcpm2_sdpa.h"
#include "timestep_embedding/voxcpm2_timestep_embedding.h"

namespace voxcpm2::runtime {

// Register the custom ncnn layers required by a given component graph.
// Must be called before net.load_param() for that component.
inline void register_component_layers(ncnn::Net& net, const std::string& name) {
    if (name == "dit_estimator") {
        register_voxcpm2_dtype_adapter(net);
        register_voxcpm2_sdpa(net);
        register_voxcpm2_timestep_embedding(net);
    } else if (is_dual_backend_component(name)) {
        register_voxcpm2_dtype_adapter(net);
    }
}

} // namespace voxcpm2::runtime
