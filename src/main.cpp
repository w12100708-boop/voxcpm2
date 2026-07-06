// The entry of VoxCPM2-NCNN CLI

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "progress.h"
#include "voxcpm2/audio_io.h"
#include "voxcpm2/synthesizer.h"
#include "voxcpm2/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <print>
#include <sstream>
#include <string>
#include <streambuf>
#include <string_view>
#include <vector>

#include <CLI/CLI.hpp>
#include <indicators/multi_progress.hpp>
#include <indicators/progress_bar.hpp>

#include <unistd.h>

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
    bool no_progress = false;
#ifdef VOXCPM2_ENABLE_PROFILE
    bool profile = false;
#endif
    bool smoke_components = false;
    bool tokenize = false;
};

bool stderr_is_tty() {
    return ::isatty(::fileno(stderr)) == 1;
}

int phase_weight_for(voxcpm2::progress::Phase phase) {
    using enum voxcpm2::progress::Phase;
    switch (phase) {
    case model_load:
        return 10;
    case input_audio:
        return 5;
    case prefix:
        return 15;
    case generation:
        return 65;
    case decode:
        return 4;
    case write:
        return 1;
    }
    return 0;
}

std::string progress_label(std::string_view label, int current, int total) {
    std::ostringstream oss;
    oss << label;
    if (total > 0) {
        oss << " " << current << "/" << total;
    }
    return oss.str();
}

std::size_t percent(double value) {
    return static_cast<std::size_t>(std::clamp(std::lround(value), 0L, 100L));
}

class CliProgress final : public voxcpm2::progress::Sink {
public:
    CliProgress(bool enabled, bool has_input_audio)
        : enabled_(enabled),
          total_weight_(100 - (has_input_audio ? 0 : phase_weight_for(voxcpm2::progress::Phase::input_audio))),
          total_bar_(
              indicators::option::BarWidth{34},
              indicators::option::PrefixText{"total   "},
              indicators::option::ShowPercentage{true},
              indicators::option::ShowElapsedTime{true},
              indicators::option::ShowRemainingTime{true},
              indicators::option::MaxProgress{100},
              indicators::option::Stream{std::cerr}),
          current_bar_(
              indicators::option::BarWidth{34},
              indicators::option::PrefixText{"current "},
              indicators::option::ShowPercentage{false},
              indicators::option::MaxProgress{101},
              indicators::option::Stream{std::cerr}),
          bars_(total_bar_, current_bar_) {
        if (enabled_) {
            previous_cout_ = std::cout.rdbuf(std::cerr.rdbuf());
            std::fputs("\033[?25l", stderr);
            std::fflush(stderr);
        }
    }

    ~CliProgress() override {
        if (enabled_) {
            if (not completed_) {
                std::cerr << '\n';
            }
            std::fputs("\033[?25h", stderr);
            std::fflush(stderr);
            if (previous_cout_ != nullptr) {
                std::cout.rdbuf(previous_cout_);
            }
        }
    }

    CliProgress(const CliProgress&) = delete;
    CliProgress& operator=(const CliProgress&) = delete;

    void begin_phase(voxcpm2::progress::Phase phase, std::string_view label, int steps) override {
        if (not enabled_) {
            return;
        }
        if (phase_started_) {
            finish_phase(phase_label_);
        }
        phase_label_ = std::string(label);
        phase_weight_ = phase_weight_for(phase);
        phase_steps_ = std::max(1, steps);
        phase_done_ = 0;
        phase_started_ = true;
        set_current(label, 0, phase_steps_);
        update_total(label);
    }

    void advance_phase(std::string_view label, int increment) override {
        if (not enabled_ or not phase_started_) {
            return;
        }
        phase_done_ = std::clamp(phase_done_ + std::max(0, increment), 0, phase_steps_);
        update_total(label);
    }

    void current(std::string_view label, int current, int total) override {
        if (not enabled_) {
            return;
        }
        set_current(label, current, total);
    }

    void finish_phase(std::string_view label) override {
        if (not enabled_ or not phase_started_) {
            return;
        }
        completed_weight_ += phase_weight_;
        phase_done_ = phase_steps_;
        phase_started_ = false;
        set_current(label, 1, 1);
        update_total(label);
    }

private:
    void set_current(std::string_view label, int current, int total) {
        const int safe_total = std::max(1, total);
        const int safe_current = std::clamp(current, 0, safe_total);
        current_bar_.set_option(indicators::option::PostfixText{progress_label(label, safe_current, safe_total)});
        bars_.set_progress<1>(percent(static_cast<double>(safe_current) * 100.0 / safe_total));
    }

    void update_total(std::string_view label) {
        double progress = static_cast<double>(completed_weight_);
        if (phase_started_) {
            progress += static_cast<double>(phase_weight_) * static_cast<double>(phase_done_) /
                        static_cast<double>(phase_steps_);
        }
        const double normalized = total_weight_ > 0 ? progress * 100.0 / static_cast<double>(total_weight_) : 100.0;
        total_bar_.set_option(indicators::option::PostfixText{std::string(label)});
        const std::size_t value = percent(normalized);
        completed_ = value >= 100;
        bars_.set_progress<0>(value);
    }

    bool enabled_ = false;
    bool completed_ = false;
    bool phase_started_ = false;
    int total_weight_ = 100;
    int completed_weight_ = 0;
    int phase_weight_ = 0;
    int phase_steps_ = 1;
    int phase_done_ = 0;
    std::string phase_label_;
    indicators::ProgressBar total_bar_;
    indicators::ProgressBar current_bar_;
    indicators::MultiProgress<indicators::ProgressBar, 2> bars_;
    std::streambuf* previous_cout_ = nullptr;
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
    app.add_flag("--no-progress", cli.no_progress, "Disable synthesis progress bars");
#ifdef VOXCPM2_ENABLE_PROFILE
    app.add_flag("--profile", cli.profile, "Print coarse synthesis timing to stderr");
#endif
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

        const bool generation_mode = not cli.smoke_components;
        const bool has_input_audio = not cli.prompt_audio.empty() or not cli.reference_audio.empty();
        const bool show_progress = generation_mode and not cli.no_progress and stderr_is_tty();
        CliProgress cli_progress(show_progress, has_input_audio);
        voxcpm2::progress::SinkScope progress_scope(show_progress ? &cli_progress : nullptr);

        voxcpm2::Synthesizer tts(voxcpm2::SynthesizerConfig{
            .model_dir = cli.model_dir,
            .use_vulkan = cli.vulkan,
#ifdef VOXCPM2_ENABLE_PROFILE
            .profile = cli.profile,
#endif
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
        const int input_audio_steps = static_cast<int>(not cli.prompt_audio.empty()) +
                                      static_cast<int>(not cli.reference_audio.empty());
        if (input_audio_steps > 0) {
            voxcpm2::progress::begin_phase(voxcpm2::progress::Phase::input_audio, "input audio", input_audio_steps);
        }
        if (not cli.prompt_audio.empty()) {
            voxcpm2::progress::current("prompt_audio", 1, input_audio_steps);
            options.prompt_audio = voxcpm2::read_audio_file(cli.prompt_audio, tts.input_sample_rate());
            voxcpm2::progress::advance_phase("prompt_audio");
        }
        if (not cli.reference_audio.empty()) {
            voxcpm2::progress::current("reference_audio", input_audio_steps, input_audio_steps);
            options.reference_audio = voxcpm2::read_audio_file(cli.reference_audio, tts.input_sample_rate());
            voxcpm2::progress::advance_phase("reference_audio");
        }
        if (input_audio_steps > 0) {
            voxcpm2::progress::finish_phase("input audio");
        }

        voxcpm2::AudioBuffer audio = tts.generate(options);
        voxcpm2::progress::begin_phase(voxcpm2::progress::Phase::write, "write", 1);
        voxcpm2::progress::current("write", 1, 1);
        voxcpm2::write_audio_file(cli.output, audio);
        voxcpm2::progress::advance_phase("write");
        voxcpm2::progress::finish_phase("write");
        std::println(stderr, "generated {} samples at {} Hz", audio.samples.size(), audio.sample_rate);
        std::println(stderr, "wrote {}", cli.output.string());
        return 0;
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    } catch (const std::exception& e) {
        std::println(stderr, "voxcpm2: {}", e.what());
        return 1;
    }
}
