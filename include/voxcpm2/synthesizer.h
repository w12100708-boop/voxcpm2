// VoxCPM2 speech synthesis public API

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "voxcpm2/audio.h"

namespace voxcpm2 {

struct SynthesisOptions {
    std::string text;
    std::string prompt_text;
    std::optional<AudioBuffer> prompt_audio;
    std::optional<AudioBuffer> reference_audio;
    int min_patches = 2;
    int inference_timesteps = 10;
    float cfg_value = 2.0f;
};

struct SynthesizerConfig {
    std::filesystem::path model_dir = "assets/voxcpm2";
    bool use_vulkan = false;
    int threads = 4;
    int vulkan_device = 0;
};

class Synthesizer {
public:
    explicit Synthesizer(SynthesizerConfig config);
    ~Synthesizer();

    Synthesizer(const Synthesizer&) = delete;
    Synthesizer& operator=(const Synthesizer&) = delete;
    Synthesizer(Synthesizer&&) noexcept;
    Synthesizer& operator=(Synthesizer&&) noexcept;

    [[nodiscard]] AudioBuffer generate(const SynthesisOptions& options) const;
    void smoke_components() const;

    [[nodiscard]] int input_sample_rate() const;
    [[nodiscard]] int output_sample_rate() const;
    [[nodiscard]] std::vector<std::string> missing_required_components() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace voxcpm2
