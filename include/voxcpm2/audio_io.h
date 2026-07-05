// FFmpeg audio input and output helpers for VoxCPM2-NCNN

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>

#include "voxcpm2/audio.h"

namespace voxcpm2 {

AudioBuffer read_audio_file(const std::filesystem::path& path, int target_sample_rate);
void write_audio_file(const std::filesystem::path& path, const AudioBuffer& audio);

} // namespace voxcpm2
