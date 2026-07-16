// OpenAI-compatible speech request validation for voxcpm2-server

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include "voxcpm2/audio_io.h"

#include <cstddef>
#include <expected>
#include <optional>
#include <string>
#include <string_view>

namespace voxcpm2::server {

inline constexpr std::size_t max_request_body_bytes = 64 * 1024;

struct SpeechRequest {
    std::string input;
    AudioFormat response_format = AudioFormat::mp3;
};

struct ApiError {
    int status = 400;
    std::string message;
    std::string type = "invalid_request_error";
    std::optional<std::string> param;
};

using SpeechRequestResult = std::expected<SpeechRequest, ApiError>;

[[nodiscard]] bool is_json_content_type(std::string_view content_type);
[[nodiscard]] SpeechRequestResult parse_speech_request(std::string_view body);
[[nodiscard]] std::string serialize_error(const ApiError& error);
[[nodiscard]] std::string serialize_health();
[[nodiscard]] int output_sample_rate(AudioFormat format);

} // namespace voxcpm2::server
