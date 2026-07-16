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
#include <exception>
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

class RequestDecodeError final : public std::exception {
public:
    explicit RequestDecodeError(ApiError error) : error_(std::move(error)) {}

    [[nodiscard]] const ApiError& error() const noexcept {
        return error_;
    }

    [[nodiscard]] const char* what() const noexcept override {
        return error_.message.c_str();
    }

private:
    ApiError error_;
};

[[noreturn]] void fail_decode(std::string message, std::optional<std::string> param = std::nullopt) {
    throw RequestDecodeError(invalid(std::move(message), std::move(param)));
}

struct VoiceInput {
    std::string id;
};

struct SpeechRequestPayload {
    std::string model;
    std::string input;
    VoiceInput voice;
    std::optional<std::string> instructions;
    AudioFormat response_format = AudioFormat::mp3;
    std::optional<double> speed;
    std::optional<std::string> stream_format;
};

struct ApiErrorBody {
    std::string message;
    std::string type;
    std::optional<std::string> param;
};

struct ApiErrorEnvelope {
    ApiErrorBody error;
};

struct HealthResponse {
    std::string status;
    std::string model;
};

template <std::size_t N>
void reject_unknown_fields(const json& object, const std::array<std::string_view, N>& allowed_fields) {
    for (const auto& [key, value] : object.items()) {
        static_cast<void>(value);
        if (std::ranges::find(allowed_fields, key) == allowed_fields.end()) {
            fail_decode("unknown field: " + key, key);
        }
    }
}

std::string required_string(
    const json& object,
    const char* field,
    std::string required_message,
    std::string type_message) {
    if (not object.contains(field)) {
        fail_decode(std::move(required_message), field);
    }
    const json& value = object.at(field);
    if (not value.is_string()) {
        fail_decode(std::move(type_message), field);
    }
    return value.get<std::string>();
}

void from_json(const json& value, VoiceInput& voice) {
    if (value.is_string()) {
        voice.id = value.get<std::string>();
        if (not voice.id.empty()) {
            return;
        }
        fail_decode("voice must not be empty", "voice");
    }
    if (value.is_object() and value.size() == 1 and value.contains("id") and value.at("id").is_string()) {
        voice.id = value.at("id").get<std::string>();
        if (not voice.id.empty()) {
            return;
        }
    }
    fail_decode("voice must be a non-empty string or an object containing a non-empty id", "voice");
}

AudioFormat decode_audio_format(const json& value) {
    if (not value.is_string()) {
        fail_decode("response_format must be a string", "response_format");
    }
    const std::string format = value.get<std::string>();
    if (format == "mp3") {
        return AudioFormat::mp3;
    }
    if (format == "opus") {
        return AudioFormat::opus;
    }
    if (format == "aac") {
        return AudioFormat::aac;
    }
    if (format == "flac") {
        return AudioFormat::flac;
    }
    if (format == "wav") {
        return AudioFormat::wav;
    }
    if (format == "pcm") {
        return AudioFormat::pcm;
    }
    fail_decode("response_format must be one of mp3, opus, aac, flac, wav, or pcm", "response_format");
}

void from_json(const json& request, SpeechRequestPayload& payload) {
    constexpr std::array<std::string_view, 7> allowed_fields{
        "model",
        "input",
        "voice",
        "instructions",
        "response_format",
        "speed",
        "stream_format",
    };
    reject_unknown_fields(request, allowed_fields);

    payload.model = required_string(request, "model", "model is required", "model must be a string");
    payload.input = required_string(request, "input", "input is required", "input must be a string");
    if (not request.contains("voice")) {
        fail_decode("voice is required", "voice");
    }
    payload.voice = request.at("voice").get<VoiceInput>();

    if (request.contains("instructions")) {
        if (not request.at("instructions").is_string()) {
            fail_decode("instructions must be a string", "instructions");
        }
        payload.instructions = request.at("instructions").get<std::string>();
    }
    if (request.contains("response_format")) {
        payload.response_format = decode_audio_format(request.at("response_format"));
    }
    if (request.contains("speed")) {
        if (not request.at("speed").is_number()) {
            fail_decode("speed must be a number", "speed");
        }
        payload.speed = request.at("speed").get<double>();
    }
    if (request.contains("stream_format")) {
        if (not request.at("stream_format").is_string()) {
            fail_decode("stream_format must be a string", "stream_format");
        }
        payload.stream_format = request.at("stream_format").get<std::string>();
    }
}

void to_json(json& output, const ApiErrorBody& error) {
    output = json{
        {"message", error.message},
        {"type", error.type},
        {"param", error.param.has_value() ? json(*error.param) : json(nullptr)},
        {"code", nullptr},
    };
}

void to_json(json& output, const ApiErrorEnvelope& envelope) {
    output = json{{"error", envelope.error}};
}

void to_json(json& output, const HealthResponse& health) {
    output = json{{"status", health.status}, {"model", health.model}};
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

SpeechRequestResult validate_request(SpeechRequestPayload payload) {
    if (payload.model != "voxcpm2") {
        return std::unexpected(invalid("only model 'voxcpm2' is available", "model"));
    }
    if (payload.input.empty()) {
        return std::unexpected(invalid("input must not be empty", "input"));
    }
    const std::optional<std::size_t> input_length = utf8_length(payload.input);
    if (not input_length.has_value()) {
        return std::unexpected(invalid("input must be valid UTF-8", "input"));
    }
    if (*input_length > 4096) {
        return std::unexpected(invalid("input must not exceed 4096 characters", "input"));
    }
    if (payload.instructions.has_value() and not payload.instructions->empty()) {
        return std::unexpected(invalid("instructions are not supported by voxcpm2", "instructions"));
    }
    if (payload.speed.has_value() and (not std::isfinite(*payload.speed) or *payload.speed != 1.0)) {
        return std::unexpected(invalid("only speed 1.0 is supported by voxcpm2", "speed"));
    }
    if (payload.stream_format.has_value() and *payload.stream_format != "audio") {
        return std::unexpected(invalid("only stream_format 'audio' is supported by voxcpm2", "stream_format"));
    }
    return SpeechRequest{
        .input = std::move(payload.input),
        .response_format = payload.response_format,
    };
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
        return std::unexpected(invalid("request body exceeds 64 KiB", std::nullopt, 413));
    }
    const json request = json::parse(body, nullptr, false);
    if (request.is_discarded()) {
        return std::unexpected(invalid("request body is not valid JSON"));
    }
    if (not request.is_object()) {
        return std::unexpected(invalid("request body must be a JSON object"));
    }

    try {
        return validate_request(request.get<SpeechRequestPayload>());
    } catch (const RequestDecodeError& error) {
        return std::unexpected(error.error());
    }
}

std::string serialize_error(const ApiError& error) {
    return json(ApiErrorEnvelope{
                    .error = ApiErrorBody{
                        .message = error.message,
                        .type = error.type,
                        .param = error.param,
                    },
                })
        .dump();
}

std::string serialize_health() {
    return json(HealthResponse{.status = "ok", .model = "voxcpm2"}).dump();
}

int output_sample_rate(AudioFormat format) {
    return format == AudioFormat::pcm ? 24000 : 0;
}

} // namespace voxcpm2::server
