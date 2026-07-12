// Internal runtime profiler for VoxCPM2 synthesis

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "profile.h"

#ifdef VOXCPM2_ENABLE_PROFILE

#include <algorithm>
#include <chrono>
#include <print>
#include <utility>

namespace voxcpm2::runtime {

Profile::Scope::Scope(Profile& profile, std::string_view label) : profile_(profile.enabled() ? &profile : nullptr) {
    if (profile_ != nullptr) {
        label_ = label;
        start_ = Clock::now();
    }
}

Profile::Scope::~Scope() {
    if (profile_ == nullptr) {
        return;
    }
    const double elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
    profile_->record(label_, elapsed_ms);
}

Profile::Scope::Scope(Scope&& other) noexcept
    : profile_(std::exchange(other.profile_, nullptr)), label_(std::move(other.label_)), start_(other.start_) {}

Profile::Profile(bool enabled) : enabled_(enabled) {
    if (enabled_) {
        start_ = Clock::now();
    }
}

Profile::Scope Profile::scope(std::string_view label) {
    return Scope(*this, label);
}

void Profile::count(std::string_view label, std::uint64_t increment) {
    if (not enabled_) {
        return;
    }
    auto it = std::ranges::find(counters_, label, &Counter::label);
    if (it == counters_.end()) {
        counters_.push_back(Counter{.label = std::string(label), .value = increment});
        return;
    }
    it->value += increment;
}

void Profile::report(std::string_view total_label, double audio_seconds) const {
    if (not enabled_) {
        return;
    }
    for (const Timing& timing : timings_) {
        const double average_ms = timing.calls == 0 ? 0.0 : timing.total_ms / static_cast<double>(timing.calls);
        std::println(
            stderr,
            "[profile] {:<24} count={:<4} total={:>9.2f} ms avg={:>9.2f} ms",
            timing.label,
            timing.calls,
            timing.total_ms,
            average_ms);
    }
    for (const Counter& counter : counters_) {
        std::println(stderr, "[profile] {:<24} count={}", counter.label, counter.value);
    }
    const double total_ms = std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
    std::println(stderr, "[profile] {:<24} total={:>9.2f} ms", total_label, total_ms);
    if (audio_seconds > 0.0) {
        std::println(stderr, "[profile] {:<24} value={:.3f}", "rtf", total_ms / 1000.0 / audio_seconds);
    }
}

bool Profile::enabled() const {
    return enabled_;
}

void Profile::record(std::string_view label, double elapsed_ms) {
    auto it = std::ranges::find(timings_, label, &Timing::label);
    if (it == timings_.end()) {
        timings_.push_back(Timing{.label = std::string(label), .calls = 1, .total_ms = elapsed_ms});
        return;
    }
    ++it->calls;
    it->total_ms += elapsed_ms;
}

} // namespace voxcpm2::runtime

#endif
