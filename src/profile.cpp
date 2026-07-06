// Optional internal coarse profiler for VoxCPM2 runtime

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "profile.h"

#ifdef VOXCPM2_ENABLE_PROFILE

#include <chrono>
#include <print>

namespace voxcpm2::runtime {

ProfileSpan::ProfileSpan(bool enabled) : enabled_(enabled), start_(Clock::now()), last_(start_) {}

void ProfileSpan::mark(std::string_view label) {
    if (not enabled_) {
        return;
    }
    const auto now = Clock::now();
    const double delta_ms = std::chrono::duration<double, std::milli>(now - last_).count();
    const double total_ms = std::chrono::duration<double, std::milli>(now - start_).count();
    std::println(stderr, "[profile] {:<24} {:>9.2f} ms total {:>9.2f} ms", label, delta_ms, total_ms);
    last_ = now;
}

} // namespace voxcpm2::runtime

#endif
