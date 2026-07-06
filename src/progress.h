// Internal progress reporting hooks for VoxCPM2-NCNN

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>

namespace voxcpm2::progress {

enum class Phase {
    model_load,
    input_audio,
    prefix,
    generation,
    decode,
    write,
};

class Sink {
public:
    virtual ~Sink() = default;

    virtual void begin_phase(Phase phase, std::string_view label, int steps) = 0;
    virtual void advance_phase(std::string_view label, int increment) = 0;
    virtual void current(std::string_view label, int current, int total) = 0;
    virtual void finish_phase(std::string_view label) = 0;
};

class SinkScope {
public:
    explicit SinkScope(Sink* sink) noexcept;
    ~SinkScope();

    SinkScope(const SinkScope&) = delete;
    SinkScope& operator=(const SinkScope&) = delete;
    SinkScope(SinkScope&& other) noexcept;
    SinkScope& operator=(SinkScope&& other) noexcept;

private:
    Sink* previous_ = nullptr;
    bool active_ = false;
};

[[nodiscard]] Sink* current_sink() noexcept;
void begin_phase(Phase phase, std::string_view label, int steps);
void advance_phase(std::string_view label, int increment = 1);
void current(std::string_view label, int current, int total);
void finish_phase(std::string_view label);

} // namespace voxcpm2::progress
