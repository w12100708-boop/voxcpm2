// Internal progress reporting hooks implementation

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "progress.h"

#include <utility>

namespace voxcpm2::progress {
namespace {

thread_local Sink* active_sink = nullptr;

} // namespace

SinkScope::SinkScope(Sink* sink) noexcept : previous_(active_sink), active_(true) {
    active_sink = sink;
}

SinkScope::~SinkScope() {
    if (active_) {
        active_sink = previous_;
    }
}

SinkScope::SinkScope(SinkScope&& other) noexcept : previous_(std::exchange(other.previous_, nullptr)),
                                                   active_(std::exchange(other.active_, false)) {}

SinkScope& SinkScope::operator=(SinkScope&& other) noexcept {
    if (this != &other) {
        if (active_) {
            active_sink = previous_;
        }
        previous_ = std::exchange(other.previous_, nullptr);
        active_ = std::exchange(other.active_, false);
    }
    return *this;
}

Sink* current_sink() noexcept {
    return active_sink;
}

void begin_phase(Phase phase, std::string_view label, int steps) {
    if (active_sink != nullptr) {
        active_sink->begin_phase(phase, label, steps);
    }
}

void advance_phase(std::string_view label, int increment) {
    if (active_sink != nullptr) {
        active_sink->advance_phase(label, increment);
    }
}

void current(std::string_view label, int current_value, int total) {
    if (active_sink != nullptr) {
        active_sink->current(label, current_value, total);
    }
}

void finish_phase(std::string_view label) {
    if (active_sink != nullptr) {
        active_sink->finish_phase(label);
    }
}

} // namespace voxcpm2::progress
