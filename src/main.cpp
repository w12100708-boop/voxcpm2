// The entry of VoxCPM2-NCNN CLI

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "voxcpm2/audio_io.h"
#include "voxcpm2/synthesizer.h"
#include "voxcpm2/tokenizer.h"

#include <exception>
#include <filesystem>
#include <print>
#include <string>

#include <CLI/CLI.hpp>

namespace {

struct CliOptions {
    std::filesystem::path model_dir = "assets/voxcpm2";
    std::string text;
    std::string prompt;
    std::filesystem::path prompt_audio;
    std::filesystem::path reference_audio;
    std::filesystem::path output = "out.wav";
    int min_patches = 2;
    int timesteps = 10;
    float cfg_value = 2.0f;
    int threads = 4;
    int vulkan_device = 0;
    bool vulkan = false;
    bool smoke_components = false;
    bool tokenize = false;
};

} // namespace

int main(int argc, char** argv) {
    CliOptions cli;

    CLI::App app{"VoxCPM2 TTS on ncnn"};
    app.add_option("-m,--model", cli.model_dir, "VoxCPM2 ncnn asset directory")->capture_default_str();
    app.add_option("-t,--text", cli.text, "Target text to synthesize");
    app.add_option("-o,--output", cli.output, "Output audio path. Format is inferred from extension")->capture_default_str();
    app.add_option("--prompt", cli.prompt, "Prompt text for continuation");
    app.add_option("--prompt-audio", cli.prompt_audio, "Prompt audio path for continuation");
    app.add_option("--reference-audio", cli.reference_audio, "Reference audio path for voice cloning");
    app.add_option("--min-patches", cli.min_patches, "Minimum generated latent patches before honoring stop token")
        ->capture_default_str();
    app.add_option("--timesteps", cli.timesteps, "CFM Euler steps per generated patch")->capture_default_str();
    app.add_option("--cfg-value", cli.cfg_value, "Classifier-free guidance value")->capture_default_str();
    app.add_option("--threads", cli.threads, "CPU worker threads")->capture_default_str();
    app.add_option("--vulkan-device", cli.vulkan_device, "Vulkan device index")->capture_default_str();
    app.add_flag("--vulkan", cli.vulkan, "Enable Vulkan compute");
    app.add_flag("--smoke-components", cli.smoke_components, "Run exported component smoke checks");
    app.add_flag("--tokenize", cli.tokenize, "Print token ids for --text and exit");

    try {
        app.parse(argc, argv);

        if (cli.tokenize) {
            if (cli.text.empty()) [[unlikely]] {
                throw std::runtime_error("--tokenize requires --text");
            }
            const voxcpm2::Tokenizer tokenizer = voxcpm2::Tokenizer::from_file(cli.model_dir / "tokenizer.json");
            const std::vector<int> ids = tokenizer.encode(cli.text);
            for (std::size_t i = 0; i < ids.size(); ++i) {
                if (i != 0) {
                    std::print(" ");
                }
                std::print("{}", ids[i]);
            }
            std::println("");
            return 0;
        }

        voxcpm2::Synthesizer tts(voxcpm2::SynthesizerConfig{
            .model_dir = cli.model_dir,
            .use_vulkan = cli.vulkan,
            .threads = cli.threads,
            .vulkan_device = cli.vulkan_device,
        });

        if (cli.smoke_components) {
            tts.smoke_components();
            return 0;
        }
        if (cli.text.empty()) [[unlikely]] {
            throw std::runtime_error("--text is required unless --smoke-components is used");
        }
        if (not cli.prompt.empty() and cli.prompt_audio.empty()) [[unlikely]] {
            throw std::runtime_error("--prompt requires --prompt-audio");
        }

        voxcpm2::SynthesisOptions options;
        options.text = cli.text;
        options.prompt_text = cli.prompt;
        options.min_patches = cli.min_patches;
        options.inference_timesteps = cli.timesteps;
        options.cfg_value = cli.cfg_value;
        if (not cli.prompt_audio.empty()) {
            options.prompt_audio = voxcpm2::read_audio_file(cli.prompt_audio, tts.input_sample_rate());
        }
        if (not cli.reference_audio.empty()) {
            options.reference_audio = voxcpm2::read_audio_file(cli.reference_audio, tts.input_sample_rate());
        }

        voxcpm2::AudioBuffer audio = tts.generate(options);
        std::println(stderr, "generated {} samples at {} Hz", audio.samples.size(), audio.sample_rate);
        voxcpm2::write_audio_file(cli.output, audio);
        std::println(stderr, "wrote {}", cli.output.string());
        return 0;
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    } catch (const std::exception& e) {
        std::println(stderr, "voxcpm2: {}", e.what());
        return 1;
    }
}
