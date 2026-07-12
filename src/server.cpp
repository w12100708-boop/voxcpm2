// OpenAI-compatible resident inference server for VoxCPM2-NCNN

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "server_api.h"
#include "voxcpm2/audio_io.h"
#include "voxcpm2/synthesizer.h"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <mutex>
#include <optional>
#include <print>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <CLI/CLI.hpp>
#include <crow.h>

namespace {

struct ServerOptions {
    std::filesystem::path model_dir = "assets/voxcpm2";
    std::string host = "127.0.0.1";
    int port = 8000;
    int min_patches = 2;
    int timesteps = 10;
    float cfg_value = 2.0f;
    int threads = 4;
    int vulkan_device = 0;
    bool vulkan = false;
#ifdef VOXCPM2_ENABLE_PROFILE
    bool profile = false;
#endif
};

crow::response error_response(const voxcpm2::server::ApiError& error) {
    crow::response response(error.status, voxcpm2::server::serialize_error(error));
    response.set_header("Content-Type", "application/json");
    return response;
}

crow::response internal_error_response() {
    return error_response(voxcpm2::server::ApiError{
        .status = 500,
        .message = "speech generation failed",
        .type = "server_error",
        .param = std::nullopt,
    });
}

std::string binary_body(const std::vector<std::uint8_t>& bytes) {
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

} // namespace

int main(int argc, char** argv) {
    ServerOptions cli;
    CLI::App app_cli{"Resident VoxCPM2 TTS server with an OpenAI-compatible Speech API"};
    app_cli.add_option("-m,--model", cli.model_dir, "VoxCPM2 ncnn asset directory")->capture_default_str();
    app_cli.add_option("--host", cli.host, "Listen address")->capture_default_str();
    app_cli.add_option("-p,--port", cli.port, "Listen port")
        ->check(CLI::Range(1, 65535))
        ->capture_default_str();
    app_cli.add_option("--min-patches", cli.min_patches, "Upstream min_len threshold; stop is honored when step > value")
        ->check(CLI::NonNegativeNumber)
        ->capture_default_str();
    app_cli.add_option("--timesteps", cli.timesteps, "CFM Euler steps per generated patch")
        ->check(CLI::PositiveNumber)
        ->capture_default_str();
    app_cli.add_option("--cfg-value", cli.cfg_value, "Classifier-free guidance value")->capture_default_str();
    app_cli.add_option("--threads", cli.threads, "CPU inference threads")
        ->check(CLI::PositiveNumber)
        ->capture_default_str();
    app_cli.add_option("--vulkan-device", cli.vulkan_device, "Vulkan device index")->capture_default_str();
    app_cli.add_flag("--vulkan", cli.vulkan, "Enable Vulkan compute");
#ifdef VOXCPM2_ENABLE_PROFILE
    app_cli.add_flag("--profile", cli.profile, "Print synthesis timing summary to stderr");
#endif

    try {
        app_cli.parse(argc, argv);

        std::println(stderr, "loading VoxCPM2 model from {}", cli.model_dir.string());
        voxcpm2::Synthesizer synthesizer(voxcpm2::SynthesizerConfig{
            .model_dir = cli.model_dir,
            .use_vulkan = cli.vulkan,
#ifdef VOXCPM2_ENABLE_PROFILE
            .profile = cli.profile,
#endif
            .threads = cli.threads,
            .vulkan_device = cli.vulkan_device,
        });
        const std::vector<std::string> missing_components = synthesizer.missing_required_components();
        if (not missing_components.empty()) {
            std::string message = "VoxCPM2 assets are incomplete. Missing ncnn components:";
            for (const std::string& component : missing_components) {
                message += " " + component;
            }
            throw std::runtime_error(message);
        }
        std::mutex inference_mutex;

        crow::SimpleApp server;
        CROW_ROUTE(server, "/healthz")
        ([] {
            crow::response response(200, R"({"status":"ok","model":"voxcpm2"})");
            response.set_header("Content-Type", "application/json");
            return response;
        });

        CROW_ROUTE(server, "/v1/audio/speech")
            .methods(crow::HTTPMethod::POST)([&](const crow::request& request) {
                if (not voxcpm2::server::is_json_content_type(request.get_header_value("Content-Type"))) {
                    return error_response(voxcpm2::server::ApiError{
                        .status = 415,
                        .message = "Content-Type must be application/json",
                        .type = "invalid_request_error",
                        .param = std::nullopt,
                    });
                }

                const voxcpm2::server::SpeechRequestResult parsed =
                    voxcpm2::server::parse_speech_request(request.body);
                if (const auto* error = std::get_if<voxcpm2::server::ApiError>(&parsed); error != nullptr) {
                    return error_response(*error);
                }
                const auto& speech = std::get<voxcpm2::server::SpeechRequest>(parsed);

                try {
                    voxcpm2::SynthesisOptions options;
                    options.text = speech.input;
                    options.min_patches = cli.min_patches;
                    options.inference_timesteps = cli.timesteps;
                    options.cfg_value = cli.cfg_value;

                    voxcpm2::AudioBuffer audio;
                    {
                        const std::lock_guard lock(inference_mutex);
                        audio = synthesizer.generate(options);
                    }
                    const std::vector<std::uint8_t> encoded = voxcpm2::encode_audio(
                        audio,
                        speech.response_format,
                        voxcpm2::server::output_sample_rate(speech.response_format));
                    crow::response response(200, binary_body(encoded));
                    response.set_header("Content-Type", "application/octet-stream");
                    return response;
                } catch (const std::exception& error) {
                    std::println(stderr, "voxcpm2-server: synthesis request failed: {}", error.what());
                    return internal_error_response();
                }
            });

        std::println(
            stderr,
            "voxcpm2-server listening on http://{}:{}/v1/audio/speech ({})",
            cli.host,
            cli.port,
            cli.vulkan ? "Vulkan" : "CPU");
        server.bindaddr(cli.host)
            .port(static_cast<std::uint16_t>(cli.port))
            .multithreaded()
            .run();
        return 0;
    } catch (const CLI::ParseError& error) {
        return app_cli.exit(error);
    } catch (const std::exception& error) {
        std::println(stderr, "voxcpm2-server: {}", error.what());
        return 1;
    }
}
