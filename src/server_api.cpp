// OpenAI-compatible speech request validation for voxcpm2-server

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "server_api.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

namespace voxcpm2::server {
namespace {

using json = nlohmann::json;

ApiError invalid(std::string message, std::optional<std::string> param = std::nullopt, int status = 400) {
    return ApiError{
        .status = status,
        .message = std::move(message),
        .type = "invalid_request_error",
        .param = std::move(param),
    };
}

std::optional<std::size_t> utf8_length(std::string_view text) {
    std::size_t length = 0;
    for (std::size_t i = 0; i < text.size();) {
        const auto first = static_cast<std::uint8_t>(text[i]);
        std::size_t width = 0;
        if (first <= 0x7f) {
            width = 1;
        } else if (first >= 0xc2 and first <= 0xdf) {
            width = 2;
        } else if (first >= 0xe0 and first <= 0xef) {
            width = 3;
        } else if (first >= 0xf0 and first <= 0xf4) {
            width = 4;
        } else {
            return std::nullopt;
        }
        if (i + width > text.size()) {
            return std::nullopt;
        }
        for (std::size_t j = 1; j < width; ++j) {
            const auto continuation = static_cast<std::uint8_t>(text[i + j]);
            if ((continuation & 0xc0) != 0x80) {
                return std::nullopt;
            }
        }
        if (width == 3) {
            const auto second = static_cast<std::uint8_t>(text[i + 1]);
            if ((first == 0xe0 and second < 0xa0) or (first == 0xed and second >= 0xa0)) {
                return std::nullopt;
            }
        } else if (width == 4) {
            const auto second = static_cast<std::uint8_t>(text[i + 1]);
            if ((first == 0xf0 and second < 0x90) or (first == 0xf4 and second >= 0x90)) {
                return std::nullopt;
            }
        }
        i += width;
        ++length;
    }
    return length;
}

std::optional<AudioFormat> parse_audio_format(const json& request, ApiError& error) {
    if (not request.contains("response_format")) {
        return AudioFormat::mp3;
    }
    if (not request["response_format"].is_string()) {
        error = invalid("response_format must be a string", "response_format");
        return std::nullopt;
    }
    const std::string value = request["response_format"].get<std::string>();
    if (value == "mp3") {
        return AudioFormat::mp3;
    }
    if (value == "opus") {
        return AudioFormat::opus;
    }
    if (value == "aac") {
        return AudioFormat::aac;
    }
    if (value == "flac") {
        return AudioFormat::flac;
    }
    if (value == "wav") {
        return AudioFormat::wav;
    }
    if (value == "pcm") {
        return AudioFormat::pcm;
    }
    error = invalid("response_format must be one of mp3, opus, aac, flac, wav, or pcm", "response_format");
    return std::nullopt;
}

std::optional<ApiError> validate_voice(const json& request) {
    if (not request.contains("voice")) {
        return invalid("voice is required", "voice");
    }
    const json& voice = request["voice"];
    if (voice.is_string()) {
        if (voice.get_ref<const std::string&>().empty()) {
            return invalid("voice must not be empty", "voice");
        }
        return std::nullopt;
    }
    if (voice.is_object() and voice.size() == 1 and voice.contains("id") and voice["id"].is_string() and
        not voice["id"].get_ref<const std::string&>().empty()) {
        return std::nullopt;
    }
    return invalid("voice must be a non-empty string or an object containing a non-empty id", "voice");
}

} // namespace

bool is_json_content_type(std::string_view content_type) {
    const std::size_t separator = content_type.find(';');
    content_type = content_type.substr(0, separator);
    while (not content_type.empty() and std::isspace(static_cast<unsigned char>(content_type.front())) != 0) {
        content_type.remove_prefix(1);
    }
    while (not content_type.empty() and std::isspace(static_cast<unsigned char>(content_type.back())) != 0) {
        content_type.remove_suffix(1);
    }
    constexpr std::string_view expected = "application/json";
    if (content_type.size() != expected.size()) {
        return false;
    }
    for (std::size_t i = 0; i < content_type.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(content_type[i])) != expected[i]) {
            return false;
        }
    }
    return true;
}

SpeechRequestResult parse_speech_request(std::string_view body) {
    if (body.size() > max_request_body_bytes) {
        return invalid("request body exceeds 64 KiB", std::nullopt, 413);
    }
    const json request = json::parse(body, nullptr, false);
    if (request.is_discarded()) {
        return invalid("request body is not valid JSON");
    }
    if (not request.is_object()) {
        return invalid("request body must be a JSON object");
    }

    constexpr std::array<std::string_view, 7> allowed_fields{
        "model",
        "input",
        "voice",
        "instructions",
        "response_format",
        "speed",
        "stream_format",
    };
    for (const auto& [key, value] : request.items()) {
        static_cast<void>(value);
        if (std::ranges::find(allowed_fields, key) == allowed_fields.end()) {
            return invalid("unknown field: " + key, key);
        }
    }

    if (not request.contains("model")) {
        return invalid("model is required", "model");
    }
    if (not request["model"].is_string()) {
        return invalid("model must be a string", "model");
    }
    if (request["model"].get_ref<const std::string&>() != "voxcpm2") {
        return invalid("only model 'voxcpm2' is available", "model");
    }

    if (not request.contains("input")) {
        return invalid("input is required", "input");
    }
    if (not request["input"].is_string()) {
        return invalid("input must be a string", "input");
    }
    const std::string input = request["input"].get<std::string>();
    if (input.empty()) {
        return invalid("input must not be empty", "input");
    }
    const std::optional<std::size_t> input_length = utf8_length(input);
    if (not input_length.has_value()) {
        return invalid("input must be valid UTF-8", "input");
    }
    if (*input_length > 4096) {
        return invalid("input must not exceed 4096 characters", "input");
    }

    if (const std::optional<ApiError> voice_error = validate_voice(request); voice_error.has_value()) {
        return *voice_error;
    }

    if (request.contains("instructions")) {
        if (not request["instructions"].is_string()) {
            return invalid("instructions must be a string", "instructions");
        }
        if (not request["instructions"].get_ref<const std::string&>().empty()) {
            return invalid("instructions are not supported by voxcpm2", "instructions");
        }
    }

    if (request.contains("speed")) {
        if (not request["speed"].is_number()) {
            return invalid("speed must be a number", "speed");
        }
        const double speed = request["speed"].get<double>();
        if (not std::isfinite(speed) or speed != 1.0) {
            return invalid("only speed 1.0 is supported by voxcpm2", "speed");
        }
    }

    if (request.contains("stream_format")) {
        if (not request["stream_format"].is_string()) {
            return invalid("stream_format must be a string", "stream_format");
        }
        if (request["stream_format"].get_ref<const std::string&>() != "audio") {
            return invalid("only stream_format 'audio' is supported by voxcpm2", "stream_format");
        }
    }

    ApiError format_error;
    const std::optional<AudioFormat> response_format = parse_audio_format(request, format_error);
    if (not response_format.has_value()) {
        return format_error;
    }
    return SpeechRequest{
        .input = input,
        .response_format = *response_format,
    };
}

std::string serialize_error(const ApiError& error) {
    const json body = {
        {"error",
         {
             {"message", error.message},
             {"type", error.type},
             {"param", error.param.has_value() ? json(*error.param) : json(nullptr)},
             {"code", nullptr},
         }},
    };
    return body.dump();
}

int output_sample_rate(AudioFormat format) {
    return format == AudioFormat::pcm ? 24000 : 0;
}

} // namespace voxcpm2::server
