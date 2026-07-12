// OpenAI-compatible speech request validation tests

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "server_api.h"

#include <array>
#include <stdexcept>
#include <string>
#include <variant>

namespace {

void require(bool condition, const char* message) {
    if (not condition) {
        throw std::runtime_error(message);
    }
}

voxcpm2::server::SpeechRequest require_request(const std::string& body) {
    const voxcpm2::server::SpeechRequestResult result = voxcpm2::server::parse_speech_request(body);
    const auto* request = std::get_if<voxcpm2::server::SpeechRequest>(&result);
    require(request != nullptr, "request should be accepted");
    return *request;
}

voxcpm2::server::ApiError require_error(const std::string& body) {
    const voxcpm2::server::SpeechRequestResult result = voxcpm2::server::parse_speech_request(body);
    const auto* error = std::get_if<voxcpm2::server::ApiError>(&result);
    require(error != nullptr, "request should be rejected");
    return *error;
}

std::string request_with(std::string field) {
    return "{\"model\":\"voxcpm2\",\"input\":\"你好\",\"voice\":\"alloy\"," + field + "}";
}

} // namespace

int main() {
    using voxcpm2::AudioFormat;

    require(voxcpm2::server::is_json_content_type("application/json"), "application/json should be accepted");
    require(voxcpm2::server::is_json_content_type("Application/JSON; charset=utf-8"), "JSON content type should be case-insensitive");
    require(not voxcpm2::server::is_json_content_type("text/json"), "text/json should be rejected");

    const auto default_request = require_request(
        R"({"model":"voxcpm2","input":"你好","voice":"alloy"})");
    require(default_request.input == "你好", "input should be preserved");
    require(default_request.response_format == AudioFormat::mp3, "MP3 should be the default format");

    require_request(R"({"model":"voxcpm2","input":"hello","voice":{"id":"voice_1"}})");
    require_request(request_with("\"instructions\":\"\",\"speed\":1.0,\"stream_format\":\"audio\""));

    struct FormatCase {
        const char* name;
        AudioFormat format;
    };
    constexpr std::array formats{
        FormatCase{"mp3", AudioFormat::mp3},
        FormatCase{"opus", AudioFormat::opus},
        FormatCase{"aac", AudioFormat::aac},
        FormatCase{"flac", AudioFormat::flac},
        FormatCase{"wav", AudioFormat::wav},
        FormatCase{"pcm", AudioFormat::pcm},
    };
    for (const FormatCase& format : formats) {
        const auto parsed = require_request(request_with(
            std::string("\"response_format\":\"") + format.name + "\""));
        require(parsed.response_format == format.format, "response format mapping mismatch");
    }

    std::string max_input;
    for (int i = 0; i < 4096; ++i) {
        max_input += "你";
    }
    require_request(
        std::string("{\"model\":\"voxcpm2\",\"input\":\"") + max_input + "\",\"voice\":\"alloy\"}");
    max_input += "你";
    require(require_error(
                std::string("{\"model\":\"voxcpm2\",\"input\":\"") + max_input + "\",\"voice\":\"alloy\"}")
                .param == "input",
            "4097 Unicode characters should be rejected");

    require_error("not json");
    require_error("[]");
    require_error(R"({"input":"hello","voice":"alloy"})");
    require_error(R"({"model":"other","input":"hello","voice":"alloy"})");
    require_error(R"({"model":"voxcpm2","voice":"alloy"})");
    require_error(R"({"model":"voxcpm2","input":"","voice":"alloy"})");
    require_error(R"({"model":"voxcpm2","input":"hello"})");
    require_error(R"({"model":"voxcpm2","input":"hello","voice":{"id":""}})");
    require_error(request_with("\"unknown\":true"));
    require_error(request_with("\"instructions\":\"speak softly\""));
    require_error(request_with("\"speed\":1.25"));
    require_error(request_with("\"stream_format\":\"sse\""));
    require_error(request_with("\"response_format\":\"ogg\""));

    const std::string oversized(voxcpm2::server::max_request_body_bytes + 1, 'x');
    require(require_error(oversized).status == 413, "oversized request should return 413");

    const voxcpm2::server::ApiError error{
        .status = 500,
        .message = "generation failed",
        .type = "server_error",
        .param = std::nullopt,
    };
    const std::string serialized = voxcpm2::server::serialize_error(error);
    require(serialized.find("\"server_error\"") != std::string::npos, "error type should be serialized");
    require(serialized.find("\"param\":null") != std::string::npos, "null error param should be serialized");
    require(voxcpm2::server::output_sample_rate(AudioFormat::pcm) == 24000, "PCM must be 24 kHz");
    require(voxcpm2::server::output_sample_rate(AudioFormat::wav) == 0, "container formats should keep native rate");
    return 0;
}
