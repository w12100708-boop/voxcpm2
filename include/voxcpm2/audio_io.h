// FFmpeg audio input and output helpers for VoxCPM2-NCNN

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "voxcpm2/audio.h"

namespace voxcpm2 {

enum class AudioFormat {
    mp3,
    opus,
    aac,
    flac,
    wav,
    pcm,
};

AudioBuffer read_audio_file(const std::filesystem::path& path, int target_sample_rate);
// A target sample rate of zero preserves AudioBuffer::sample_rate.
[[nodiscard]] std::vector<std::uint8_t> encode_audio(
    const AudioBuffer& audio,
    AudioFormat format,
    int target_sample_rate = 0);
void write_audio_file(const std::filesystem::path& path, const AudioBuffer& audio);

} // namespace voxcpm2
