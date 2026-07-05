// Audio buffer type used by VoxCPM2-NCNN

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <vector>

namespace voxcpm2 {

struct AudioBuffer {
    int sample_rate = 0;
    int channels = 1;
    std::vector<float> samples;

    [[nodiscard]] std::size_t frame_count() const {
        if (channels <= 0) {
            return 0;
        }
        return samples.size() / static_cast<std::size_t>(channels);
    }
};

} // namespace voxcpm2
