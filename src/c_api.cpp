// Stable C ABI for the VoxCPM2 synthesis runtime

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "voxcpm2/c_api.h"

#include "progress.h"
#include "voxcpm2/synthesizer.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

struct voxcpm2_operation {
    std::atomic_bool cancelled{false};
};

struct voxcpm2_synthesizer {
    explicit voxcpm2_synthesizer(std::unique_ptr<voxcpm2::Synthesizer> value)
        : synthesizer(std::move(value)) {}

    std::mutex state_mutex;
    std::condition_variable state_changed;
    std::mutex call_mutex;
    bool closing = false;
    std::size_t users = 0;
    voxcpm2_operation* active_operation = nullptr;
    std::unique_ptr<voxcpm2::Synthesizer> synthesizer;
};

namespace {

class Cancelled final : public std::runtime_error {
public:
    Cancelled() : std::runtime_error("VoxCPM2 operation cancelled") {}
};

class InvalidState final : public std::runtime_error {
public:
    InvalidState() : std::runtime_error("invalid or closed VoxCPM2 native handle") {}
};

char* copy_string(std::string_view value) {
    auto output = std::make_unique<char[]>(value.size() + 1);
    if (not value.empty()) {
        std::memcpy(output.get(), value.data(), value.size());
    }
    output[value.size()] = '\0';
    return output.release();
}

void initialize_error(voxcpm2_error* error) noexcept {
    if (error != nullptr) {
        error->code = VOXCPM2_STATUS_OK;
        error->message = nullptr;
    }
}

void set_error(voxcpm2_error* error, voxcpm2_status status, std::string_view message) noexcept {
    if (error == nullptr) {
        return;
    }
    error->code = status;
    try {
        error->message = copy_string(message);
    } catch (...) {
        error->message = nullptr;
    }
}

[[nodiscard]] bool is_cancelled(const voxcpm2_operation* operation) noexcept {
    return operation != nullptr and operation->cancelled.load(std::memory_order_acquire);
}

void check_cancelled(const voxcpm2_operation* operation) {
    if (is_cancelled(operation)) {
        throw Cancelled{};
    }
}

voxcpm2_progress_phase c_phase(voxcpm2::progress::Phase phase) {
    using enum voxcpm2::progress::Phase;
    switch (phase) {
    case model_load:
        return VOXCPM2_PROGRESS_MODEL_LOAD;
    case prefix:
    case input_audio:
        return VOXCPM2_PROGRESS_PREFIX;
    case generation:
        return VOXCPM2_PROGRESS_GENERATION;
    case decode:
    case write:
        return VOXCPM2_PROGRESS_DECODE;
    case smoke:
        return VOXCPM2_PROGRESS_SMOKE;
    }
    return VOXCPM2_PROGRESS_GENERATION;
}

class CProgressSink final : public voxcpm2::progress::Sink {
public:
    CProgressSink(
        voxcpm2_operation* operation,
        voxcpm2_progress_callback callback,
        void* user_data) noexcept
        : operation_(operation), callback_(callback), user_data_(user_data) {}

    void begin_phase(voxcpm2::progress::Phase phase, std::string_view label, int steps) override {
        phase_ = c_phase(phase);
        completed_ = 0;
        total_ = std::max(steps, 0);
        report(label);
    }

    void advance_phase(std::string_view label, int increment) override {
        completed_ = std::clamp(completed_ + std::max(increment, 0), 0, total_);
        report(label);
    }

    void current(std::string_view label, int, int) override {
        report(label);
    }

    void finish_phase(std::string_view label) override {
        completed_ = total_;
        report(label);
    }

private:
    void report(std::string_view label) {
        check_cancelled(operation_);
        if (callback_ == nullptr) {
            return;
        }
        const std::string stable_label(label);
        if (callback_(user_data_, phase_, stable_label.c_str(), completed_, total_) == 0) {
            if (operation_ != nullptr) {
                operation_->cancelled.store(true, std::memory_order_release);
            }
            throw Cancelled{};
        }
        check_cancelled(operation_);
    }

    voxcpm2_operation* operation_ = nullptr;
    voxcpm2_progress_callback callback_ = nullptr;
    void* user_data_ = nullptr;
    voxcpm2_progress_phase phase_ = VOXCPM2_PROGRESS_MODEL_LOAD;
    int completed_ = 0;
    int total_ = 0;
};

template <typename Callable>
voxcpm2_status guard(voxcpm2_error* error, Callable&& callable) noexcept {
    initialize_error(error);
    try {
        std::forward<Callable>(callable)();
        return VOXCPM2_STATUS_OK;
    } catch (const Cancelled& exception) {
        set_error(error, VOXCPM2_STATUS_CANCELLED, exception.what());
        return VOXCPM2_STATUS_CANCELLED;
    } catch (const InvalidState& exception) {
        set_error(error, VOXCPM2_STATUS_INVALID_STATE, exception.what());
        return VOXCPM2_STATUS_INVALID_STATE;
    } catch (const std::invalid_argument& exception) {
        set_error(error, VOXCPM2_STATUS_INVALID_ARGUMENT, exception.what());
        return VOXCPM2_STATUS_INVALID_ARGUMENT;
    } catch (const std::bad_alloc&) {
        set_error(error, VOXCPM2_STATUS_OUT_OF_MEMORY, "native allocation failed");
        return VOXCPM2_STATUS_OUT_OF_MEMORY;
    } catch (const std::exception& exception) {
        set_error(error, VOXCPM2_STATUS_RUNTIME_ERROR, exception.what());
        return VOXCPM2_STATUS_RUNTIME_ERROR;
    } catch (...) {
        set_error(error, VOXCPM2_STATUS_RUNTIME_ERROR, "unknown native exception");
        return VOXCPM2_STATUS_RUNTIME_ERROR;
    }
}

void validate_config(const voxcpm2_config* config) {
    if (config == nullptr) {
        throw std::invalid_argument("config must not be null");
    }
    if (config->model_dir_utf8 == nullptr or config->model_dir_utf8[0] == '\0') {
        throw std::invalid_argument("modelDir must not be empty");
    }
    if (config->threads <= 0) {
        throw std::invalid_argument("threads must be positive");
    }
    if (config->vulkan_device < 0) {
        throw std::invalid_argument("vulkanDevice must not be negative");
    }
}

std::optional<voxcpm2::AudioBuffer> cpp_audio(
    const voxcpm2_audio_view* view,
    int expected_sample_rate,
    std::string_view field) {
    if (view == nullptr) {
        return std::nullopt;
    }
    if (view->samples == nullptr or view->sample_count == 0) {
        throw std::invalid_argument(std::string(field) + " must not be empty");
    }
    if (view->channels != 1) {
        throw std::invalid_argument(std::string(field) + " must be mono");
    }
    if (view->sample_rate != expected_sample_rate) {
        throw std::invalid_argument(
            std::string(field) + " sample rate must equal the synthesizer input sample rate");
    }
    voxcpm2::AudioBuffer audio;
    audio.sample_rate = view->sample_rate;
    audio.channels = view->channels;
    audio.samples.assign(view->samples, view->samples + view->sample_count);
    return audio;
}

template <typename Callable>
void with_synthesizer(
    voxcpm2_synthesizer* handle,
    voxcpm2_operation* operation,
    Callable&& callable) {
    if (handle == nullptr) {
        throw InvalidState{};
    }
    check_cancelled(operation);
    {
        std::scoped_lock state_lock(handle->state_mutex);
        if (handle->closing or handle->synthesizer == nullptr) {
            throw InvalidState{};
        }
        ++handle->users;
    }

    auto release_user = [&]() noexcept {
        std::scoped_lock state_lock(handle->state_mutex);
        --handle->users;
        handle->state_changed.notify_all();
    };

    try {
        std::unique_lock call_lock(handle->call_mutex);
        {
            std::scoped_lock state_lock(handle->state_mutex);
            if (handle->closing or handle->synthesizer == nullptr) {
                throw InvalidState{};
            }
            handle->active_operation = operation;
        }
        check_cancelled(operation);
        std::forward<Callable>(callable)(*handle->synthesizer);
        {
            std::scoped_lock state_lock(handle->state_mutex);
            handle->active_operation = nullptr;
        }
    } catch (...) {
        {
            std::scoped_lock state_lock(handle->state_mutex);
            if (handle->active_operation == operation) {
                handle->active_operation = nullptr;
            }
        }
        release_user();
        throw;
    }
    release_user();
}

} // namespace

extern "C" uint32_t voxcpm2_c_abi_version(void) {
    return VOXCPM2_C_ABI_VERSION;
}

extern "C" voxcpm2_operation* voxcpm2_operation_create(void) {
    return new (std::nothrow) voxcpm2_operation{};
}

extern "C" void voxcpm2_operation_cancel(voxcpm2_operation* operation) {
    if (operation != nullptr) {
        operation->cancelled.store(true, std::memory_order_release);
    }
}

extern "C" int32_t voxcpm2_operation_is_cancelled(const voxcpm2_operation* operation) {
    return is_cancelled(operation) ? 1 : 0;
}

extern "C" void voxcpm2_operation_destroy(voxcpm2_operation* operation) {
    delete operation;
}

extern "C" voxcpm2_status voxcpm2_synthesizer_create(
    const voxcpm2_config* config,
    voxcpm2_operation* operation,
    voxcpm2_progress_callback progress,
    void* progress_user_data,
    voxcpm2_synthesizer** out_synthesizer,
    voxcpm2_error* out_error) {
    if (out_synthesizer != nullptr) {
        *out_synthesizer = nullptr;
    }
    return guard(out_error, [&]() {
        if (out_synthesizer == nullptr) {
            throw std::invalid_argument("outSynthesizer must not be null");
        }
        validate_config(config);
        check_cancelled(operation);
        CProgressSink sink(operation, progress, progress_user_data);
        voxcpm2::progress::SinkScope scope(&sink);
        auto synthesizer = std::make_unique<voxcpm2::Synthesizer>(voxcpm2::SynthesizerConfig{
            .model_dir = config->model_dir_utf8,
            .use_vulkan = config->use_vulkan != 0,
            .profile = config->profile != 0,
            .threads = config->threads,
            .vulkan_device = config->vulkan_device,
        });
        check_cancelled(operation);
        *out_synthesizer = new voxcpm2_synthesizer(std::move(synthesizer));
    });
}

extern "C" void voxcpm2_synthesizer_destroy(voxcpm2_synthesizer* synthesizer) {
    if (synthesizer == nullptr) {
        return;
    }
    std::unique_lock state_lock(synthesizer->state_mutex);
    {
        synthesizer->closing = true;
        voxcpm2_operation_cancel(synthesizer->active_operation);
    }
    synthesizer->state_changed.wait(state_lock, [synthesizer]() { return synthesizer->users == 0; });
    synthesizer->synthesizer.reset();
    state_lock.unlock();
    delete synthesizer;
}

extern "C" voxcpm2_status voxcpm2_synthesizer_generate(
    voxcpm2_synthesizer* synthesizer,
    const voxcpm2_synthesis_options* options,
    voxcpm2_operation* operation,
    voxcpm2_progress_callback progress,
    void* progress_user_data,
    voxcpm2_audio_buffer* out_audio,
    voxcpm2_error* out_error) {
    if (out_audio != nullptr) {
        *out_audio = {};
    }
    return guard(out_error, [&]() {
        if (options == nullptr) {
            throw std::invalid_argument("options must not be null");
        }
        if (out_audio == nullptr) {
            throw std::invalid_argument("outAudio must not be null");
        }
        if (options->text_utf8 == nullptr or options->text_utf8[0] == '\0') {
            throw std::invalid_argument("text must not be empty");
        }
        if (options->min_patches < 0) {
            throw std::invalid_argument("minPatches must not be negative");
        }
        if (options->inference_timesteps <= 0) {
            throw std::invalid_argument("inferenceTimesteps must be positive");
        }
        if (not std::isfinite(options->cfg_value)) {
            throw std::invalid_argument("cfgValue must be finite");
        }

        with_synthesizer(synthesizer, operation, [&](const voxcpm2::Synthesizer& value) {
            voxcpm2::SynthesisOptions cpp_options;
            cpp_options.text = options->text_utf8;
            cpp_options.prompt_text = options->prompt_text_utf8 == nullptr ? "" : options->prompt_text_utf8;
            cpp_options.prompt_audio = cpp_audio(
                options->prompt_audio,
                value.input_sample_rate(),
                "promptAudio");
            cpp_options.reference_audio = cpp_audio(
                options->reference_audio,
                value.input_sample_rate(),
                "referenceAudio");
            cpp_options.min_patches = options->min_patches;
            cpp_options.inference_timesteps = options->inference_timesteps;
            cpp_options.cfg_value = options->cfg_value;
            if (not cpp_options.prompt_text.empty() and not cpp_options.prompt_audio.has_value()) {
                throw std::invalid_argument("promptText requires promptAudio");
            }

            CProgressSink sink(operation, progress, progress_user_data);
            voxcpm2::progress::SinkScope scope(&sink);
            voxcpm2::AudioBuffer generated = value.generate(cpp_options);
            check_cancelled(operation);
            if (generated.samples.size() > std::numeric_limits<size_t>::max() / sizeof(float)) {
                throw std::length_error("generated audio is too large");
            }
            auto samples = std::make_unique<float[]>(generated.samples.size());
            if (not generated.samples.empty()) {
                std::copy(generated.samples.begin(), generated.samples.end(), samples.get());
            }
            out_audio->samples = samples.release();
            out_audio->sample_count = generated.samples.size();
            out_audio->sample_rate = generated.sample_rate;
            out_audio->channels = generated.channels;
        });
    });
}

extern "C" voxcpm2_status voxcpm2_synthesizer_smoke_components(
    voxcpm2_synthesizer* synthesizer,
    voxcpm2_operation* operation,
    voxcpm2_progress_callback progress,
    void* progress_user_data,
    voxcpm2_error* out_error) {
    return guard(out_error, [&]() {
        with_synthesizer(synthesizer, operation, [&](const voxcpm2::Synthesizer& value) {
            CProgressSink sink(operation, progress, progress_user_data);
            voxcpm2::progress::SinkScope scope(&sink);
            value.smoke_components();
            check_cancelled(operation);
        });
    });
}

extern "C" voxcpm2_status voxcpm2_synthesizer_input_sample_rate(
    voxcpm2_synthesizer* synthesizer,
    int32_t* out_sample_rate,
    voxcpm2_error* out_error) {
    if (out_sample_rate != nullptr) {
        *out_sample_rate = 0;
    }
    return guard(out_error, [&]() {
        if (out_sample_rate == nullptr) {
            throw std::invalid_argument("outSampleRate must not be null");
        }
        with_synthesizer(synthesizer, nullptr, [&](const voxcpm2::Synthesizer& value) {
            *out_sample_rate = value.input_sample_rate();
        });
    });
}

extern "C" voxcpm2_status voxcpm2_synthesizer_output_sample_rate(
    voxcpm2_synthesizer* synthesizer,
    int32_t* out_sample_rate,
    voxcpm2_error* out_error) {
    if (out_sample_rate != nullptr) {
        *out_sample_rate = 0;
    }
    return guard(out_error, [&]() {
        if (out_sample_rate == nullptr) {
            throw std::invalid_argument("outSampleRate must not be null");
        }
        with_synthesizer(synthesizer, nullptr, [&](const voxcpm2::Synthesizer& value) {
            *out_sample_rate = value.output_sample_rate();
        });
    });
}

extern "C" voxcpm2_status voxcpm2_synthesizer_missing_required_components(
    voxcpm2_synthesizer* synthesizer,
    voxcpm2_string_list* out_components,
    voxcpm2_error* out_error) {
    if (out_components != nullptr) {
        *out_components = {};
    }
    return guard(out_error, [&]() {
        if (out_components == nullptr) {
            throw std::invalid_argument("outComponents must not be null");
        }
        with_synthesizer(synthesizer, nullptr, [&](const voxcpm2::Synthesizer& value) {
            const std::vector<std::string> components = value.missing_required_components();
            auto items = std::make_unique<char*[]>(components.size());
            std::fill_n(items.get(), components.size(), nullptr);
            try {
                for (std::size_t index = 0; index < components.size(); ++index) {
                    items[index] = copy_string(components[index]);
                }
            } catch (...) {
                for (std::size_t index = 0; index < components.size(); ++index) {
                    delete[] items[index];
                }
                throw;
            }
            out_components->items = items.release();
            out_components->count = components.size();
        });
    });
}

extern "C" void voxcpm2_audio_buffer_free(voxcpm2_audio_buffer* buffer) {
    if (buffer == nullptr) {
        return;
    }
    delete[] buffer->samples;
    buffer->samples = nullptr;
    buffer->sample_count = 0;
    buffer->sample_rate = 0;
    buffer->channels = 0;
}

extern "C" void voxcpm2_string_list_free(voxcpm2_string_list* list) {
    if (list == nullptr) {
        return;
    }
    for (std::size_t index = 0; index < list->count; ++index) {
        delete[] list->items[index];
    }
    delete[] list->items;
    list->items = nullptr;
    list->count = 0;
}

extern "C" void voxcpm2_error_free(voxcpm2_error* error) {
    if (error == nullptr) {
        return;
    }
    delete[] error->message;
    initialize_error(error);
}
