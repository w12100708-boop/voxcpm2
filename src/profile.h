// Optional internal coarse profiler for VoxCPM2 runtime

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef VOXCPM2_ENABLE_PROFILE

#include <chrono>
#include <string_view>

namespace voxcpm2::runtime {

class ProfileSpan {
public:
    explicit ProfileSpan(bool enabled);

    void mark(std::string_view label);

private:
    using Clock = std::chrono::steady_clock;

    bool enabled_ = false;
    Clock::time_point start_;
    Clock::time_point last_;
};

} // namespace voxcpm2::runtime

#endif
