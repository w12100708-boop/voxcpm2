// FFmpeg in-memory audio encoding regression tests

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "voxcpm2/audio_io.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (not condition) {
        throw std::runtime_error(message);
    }
}

voxcpm2::AudioBuffer make_tone() {
    constexpr int sample_rate = 48000;
    constexpr int sample_count = sample_rate / 10;
    std::vector<float> samples(static_cast<std::size_t>(sample_count));
    for (int i = 0; i < sample_count; ++i) {
        const float phase = 2.0f * std::numbers::pi_v<float> * 440.0f * static_cast<float>(i) /
                            static_cast<float>(sample_rate);
        samples[static_cast<std::size_t>(i)] = std::sin(phase) * 0.25f;
    }
    return voxcpm2::AudioBuffer{
        .sample_rate = sample_rate,
        .channels = 1,
        .samples = std::move(samples),
    };
}

void write_bytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (not output) {
        throw std::runtime_error("failed to write encoded audio test file");
    }
}

} // namespace

int main() {
    const voxcpm2::AudioBuffer audio = make_tone();
    struct EncodedCase {
        voxcpm2::AudioFormat format;
        const char* extension;
    };
    constexpr std::array cases{
        EncodedCase{voxcpm2::AudioFormat::mp3, "mp3"},
        EncodedCase{voxcpm2::AudioFormat::opus, "opus"},
        EncodedCase{voxcpm2::AudioFormat::aac, "aac"},
        EncodedCase{voxcpm2::AudioFormat::flac, "flac"},
        EncodedCase{voxcpm2::AudioFormat::wav, "wav"},
    };

    for (const EncodedCase& test_case : cases) {
        const std::vector<std::uint8_t> encoded = voxcpm2::encode_audio(audio, test_case.format);
        require(not encoded.empty(), "encoded audio must not be empty");

        const std::filesystem::path path =
            std::filesystem::temp_directory_path() /
            (std::string("voxcpm2_audio_io_test.") + test_case.extension);
        write_bytes(path, encoded);
        const voxcpm2::AudioBuffer decoded = voxcpm2::read_audio_file(path, audio.sample_rate);
        require(decoded.sample_rate == audio.sample_rate, "decoded sample rate mismatch");
        require(decoded.channels == 1, "decoded audio must remain mono");
        require(not decoded.samples.empty(), "decoded audio must not be empty");
        std::filesystem::remove(path);
    }

    const std::vector<std::uint8_t> pcm =
        voxcpm2::encode_audio(audio, voxcpm2::AudioFormat::pcm, 24000);
    require(pcm.size() == 2400 * sizeof(std::int16_t), "24 kHz PCM byte count mismatch");

    const std::filesystem::path file_output =
        std::filesystem::temp_directory_path() / "voxcpm2_audio_io_write_test.wav";
    voxcpm2::write_audio_file(file_output, audio);
    const voxcpm2::AudioBuffer file_decoded = voxcpm2::read_audio_file(file_output, audio.sample_rate);
    require(file_decoded.frame_count() == audio.frame_count(), "file output frame count mismatch");
    std::filesystem::remove(file_output);

    bool rejected_negative_rate = false;
    try {
        static_cast<void>(voxcpm2::encode_audio(audio, voxcpm2::AudioFormat::wav, -1));
    } catch (const std::runtime_error&) {
        rejected_negative_rate = true;
    }
    require(rejected_negative_rate, "negative target sample rate must be rejected");
    return 0;
}
