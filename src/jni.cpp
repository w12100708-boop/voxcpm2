// Android JNI bindings for the VoxCPM2 synthesis runtime

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "voxcpm2/synthesizer.h"

#include <jni.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

class PendingJavaException final {};

class InvalidHandle final : public std::runtime_error {
public:
    InvalidHandle() : std::runtime_error("invalid or closed VoxCPM2 native handle") {}
};

class StringChars final {
public:
    StringChars(JNIEnv* env, jstring value) : env_(env), value_(value), chars_(env->GetStringChars(value, nullptr)) {
        if (chars_ == nullptr) {
            throw PendingJavaException{};
        }
    }

    ~StringChars() {
        env_->ReleaseStringChars(value_, chars_);
    }

    StringChars(const StringChars&) = delete;
    StringChars& operator=(const StringChars&) = delete;

    [[nodiscard]] const jchar* get() const {
        return chars_;
    }

private:
    JNIEnv* env_;
    jstring value_;
    const jchar* chars_;
};

void append_utf8(std::string& output, std::uint32_t codepoint) {
    if (codepoint <= 0x7f) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
        output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
        output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
}

std::string java_string(JNIEnv* env, jstring value, const char* field, bool nullable = false) {
    if (value == nullptr) {
        if (nullable) {
            return {};
        }
        throw std::invalid_argument(std::string(field) + " must not be null");
    }

    const jsize length = env->GetStringLength(value);
    StringChars chars(env, value);
    std::string output;
    output.reserve(static_cast<std::size_t>(length) * 3);
    for (jsize index = 0; index < length; ++index) {
        std::uint32_t codepoint = chars.get()[index];
        if (codepoint >= 0xd800 and codepoint <= 0xdbff) {
            if (++index >= length) {
                throw std::invalid_argument(std::string(field) + " contains an unpaired UTF-16 surrogate");
            }
            const std::uint32_t low = chars.get()[index];
            if (low < 0xdc00 or low > 0xdfff) {
                throw std::invalid_argument(std::string(field) + " contains an unpaired UTF-16 surrogate");
            }
            codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
        } else if (codepoint >= 0xdc00 and codepoint <= 0xdfff) {
            throw std::invalid_argument(std::string(field) + " contains an unpaired UTF-16 surrogate");
        }
        append_utf8(output, codepoint);
    }
    return output;
}

std::optional<voxcpm2::AudioBuffer> java_audio(
    JNIEnv* env,
    jfloatArray value,
    int sample_rate,
    const char* field) {
    if (value == nullptr) {
        return std::nullopt;
    }

    const jsize length = env->GetArrayLength(value);
    if (length <= 0) {
        throw std::invalid_argument(std::string(field) + " must not be empty");
    }

    voxcpm2::AudioBuffer audio;
    audio.sample_rate = sample_rate;
    audio.channels = 1;
    audio.samples.resize(static_cast<std::size_t>(length));
    env->GetFloatArrayRegion(value, 0, length, audio.samples.data());
    if (env->ExceptionCheck()) {
        throw PendingJavaException{};
    }
    return audio;
}

jfloatArray java_audio(JNIEnv* env, const voxcpm2::AudioBuffer& audio) {
    if (audio.channels != 1) {
        throw std::runtime_error("VoxCPM2 JNI output must be mono");
    }
    if (audio.samples.size() > static_cast<std::size_t>(std::numeric_limits<jsize>::max())) {
        throw std::length_error("VoxCPM2 output is too large for a Java float array");
    }

    const jsize length = static_cast<jsize>(audio.samples.size());
    jfloatArray output = env->NewFloatArray(length);
    if (output == nullptr) {
        throw PendingJavaException{};
    }
    if (length > 0) {
        env->SetFloatArrayRegion(output, 0, length, audio.samples.data());
        if (env->ExceptionCheck()) {
            env->DeleteLocalRef(output);
            throw PendingJavaException{};
        }
    }
    return output;
}

jobjectArray java_strings(JNIEnv* env, const std::vector<std::string>& values) {
    if (values.size() > static_cast<std::size_t>(std::numeric_limits<jsize>::max())) {
        throw std::length_error("too many strings for a Java array");
    }
    jclass string_class = env->FindClass("java/lang/String");
    if (string_class == nullptr) {
        throw PendingJavaException{};
    }
    jobjectArray output = env->NewObjectArray(static_cast<jsize>(values.size()), string_class, nullptr);
    env->DeleteLocalRef(string_class);
    if (output == nullptr) {
        throw PendingJavaException{};
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
        jstring value = env->NewStringUTF(values[index].c_str());
        if (value == nullptr) {
            env->DeleteLocalRef(output);
            throw PendingJavaException{};
        }
        env->SetObjectArrayElement(output, static_cast<jsize>(index), value);
        env->DeleteLocalRef(value);
        if (env->ExceptionCheck()) {
            env->DeleteLocalRef(output);
            throw PendingJavaException{};
        }
    }
    return output;
}

void throw_java(JNIEnv* env, const char* class_name, const char* message) noexcept {
    if (env->ExceptionCheck()) {
        return;
    }
    jclass exception_class = env->FindClass(class_name);
    if (exception_class == nullptr) {
        return;
    }
    env->ThrowNew(exception_class, message);
    env->DeleteLocalRef(exception_class);
}

template <typename Result, typename Callable>
Result jni_guard(JNIEnv* env, Result fallback, Callable&& callable) noexcept {
    try {
        return std::forward<Callable>(callable)();
    } catch (const PendingJavaException&) {
        return fallback;
    } catch (const InvalidHandle& error) {
        throw_java(env, "java/lang/IllegalStateException", error.what());
    } catch (const std::invalid_argument& error) {
        throw_java(env, "java/lang/IllegalArgumentException", error.what());
    } catch (const std::bad_alloc&) {
        throw_java(env, "java/lang/OutOfMemoryError", "native allocation failed");
    } catch (const std::exception& error) {
        throw_java(env, "java/lang/RuntimeException", error.what());
    } catch (...) {
        throw_java(env, "java/lang/RuntimeException", "unknown native exception");
    }
    return fallback;
}

template <typename Callable>
void jni_guard(JNIEnv* env, Callable&& callable) noexcept {
    static_cast<void>(jni_guard(env, false, [&]() {
        std::forward<Callable>(callable)();
        return true;
    }));
}

struct NativeSynthesizer {
    explicit NativeSynthesizer(voxcpm2::SynthesizerConfig config) : synthesizer(std::move(config)) {}

    std::mutex mutex;
    bool closing = false;
    voxcpm2::Synthesizer synthesizer;
};

std::mutex registry_mutex;
std::unordered_map<jlong, std::shared_ptr<NativeSynthesizer>> registry;
jlong next_handle = 1;

jlong add_synthesizer(std::shared_ptr<NativeSynthesizer> synthesizer) {
    std::scoped_lock lock(registry_mutex);
    const jlong first_candidate = next_handle;
    do {
        const jlong candidate = next_handle;
        next_handle = next_handle == std::numeric_limits<jlong>::max() ? 1 : next_handle + 1;
        if (not registry.contains(candidate)) {
            registry.emplace(candidate, std::move(synthesizer));
            return candidate;
        }
    } while (next_handle != first_candidate);
    throw std::runtime_error("VoxCPM2 native handle space exhausted");
}

std::shared_ptr<NativeSynthesizer> get_synthesizer(jlong handle) {
    if (handle <= 0) {
        throw InvalidHandle{};
    }
    std::scoped_lock lock(registry_mutex);
    const auto found = registry.find(handle);
    if (found == registry.end()) {
        throw InvalidHandle{};
    }
    return found->second;
}

void remove_synthesizer(jlong handle) {
    if (handle == 0) {
        return;
    }
    std::shared_ptr<NativeSynthesizer> synthesizer;
    {
        std::scoped_lock lock(registry_mutex);
        const auto found = registry.find(handle);
        if (found == registry.end()) {
            throw InvalidHandle{};
        }
        synthesizer = std::move(found->second);
        registry.erase(found);
    }
    std::scoped_lock lock(synthesizer->mutex);
    synthesizer->closing = true;
}

template <typename Callable>
decltype(auto) with_synthesizer(jlong handle, Callable&& callable) {
    const std::shared_ptr<NativeSynthesizer> synthesizer = get_synthesizer(handle);
    std::scoped_lock lock(synthesizer->mutex);
    if (synthesizer->closing) {
        throw InvalidHandle{};
    }
    return std::forward<Callable>(callable)(synthesizer->synthesizer);
}

void clear_synthesizers() {
    std::vector<std::shared_ptr<NativeSynthesizer>> synthesizers;
    {
        std::scoped_lock lock(registry_mutex);
        synthesizers.reserve(registry.size());
        for (auto& [handle, synthesizer] : registry) {
            static_cast<void>(handle);
            synthesizers.push_back(std::move(synthesizer));
        }
        registry.clear();
    }
    for (const std::shared_ptr<NativeSynthesizer>& synthesizer : synthesizers) {
        std::scoped_lock lock(synthesizer->mutex);
        synthesizer->closing = true;
    }
}

} // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    static_cast<void>(vm);
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT void JNICALL JNI_OnUnload(JavaVM*, void*) {
    clear_synthesizers();
}

extern "C" JNIEXPORT jlong JNICALL Java_top_yurin_voxcpm2_VoxCPM2_nativeCreate(
    JNIEnv* env,
    jclass,
    jstring model_dir,
    jboolean use_vulkan,
    jboolean profile,
    jint threads,
    jint vulkan_device) {
    return jni_guard(env, jlong{0}, [&]() {
        std::string model_path = java_string(env, model_dir, "modelDir");
        if (model_path.empty()) {
            throw std::invalid_argument("modelDir must not be empty");
        }
        auto synthesizer = std::make_shared<NativeSynthesizer>(voxcpm2::SynthesizerConfig{
            .model_dir = std::move(model_path),
            .use_vulkan = use_vulkan == JNI_TRUE,
            .profile = profile == JNI_TRUE,
            .threads = static_cast<int>(threads),
            .vulkan_device = static_cast<int>(vulkan_device),
        });
        return add_synthesizer(std::move(synthesizer));
    });
}

extern "C" JNIEXPORT void JNICALL Java_top_yurin_voxcpm2_VoxCPM2_nativeDestroy(
    JNIEnv* env,
    jclass,
    jlong handle) {
    jni_guard(env, [&]() { remove_synthesizer(handle); });
}

extern "C" JNIEXPORT jfloatArray JNICALL Java_top_yurin_voxcpm2_VoxCPM2_nativeGenerate(
    JNIEnv* env,
    jclass,
    jlong handle,
    jstring text,
    jstring prompt_text,
    jfloatArray prompt_audio,
    jfloatArray reference_audio,
    jint min_patches,
    jint inference_timesteps,
    jfloat cfg_value) {
    return jni_guard(env, static_cast<jfloatArray>(nullptr), [&]() {
        return with_synthesizer(handle, [&](const voxcpm2::Synthesizer& synthesizer) {
            voxcpm2::SynthesisOptions options;
            options.text = java_string(env, text, "text");
            options.prompt_text = java_string(env, prompt_text, "promptText", true);
            options.prompt_audio = java_audio(
                env,
                prompt_audio,
                synthesizer.input_sample_rate(),
                "promptAudio");
            options.reference_audio = java_audio(
                env,
                reference_audio,
                synthesizer.input_sample_rate(),
                "referenceAudio");
            options.min_patches = static_cast<int>(min_patches);
            options.inference_timesteps = static_cast<int>(inference_timesteps);
            options.cfg_value = static_cast<float>(cfg_value);
            if (options.text.empty()) {
                throw std::invalid_argument("text must not be empty");
            }
            if (not options.prompt_text.empty() and not options.prompt_audio.has_value()) {
                throw std::invalid_argument("promptText requires promptAudio");
            }
            if (options.inference_timesteps <= 0) {
                throw std::invalid_argument("inferenceTimesteps must be positive");
            }
            if (not std::isfinite(options.cfg_value)) {
                throw std::invalid_argument("cfgValue must be finite");
            }
            return java_audio(env, synthesizer.generate(options));
        });
    });
}

extern "C" JNIEXPORT void JNICALL Java_top_yurin_voxcpm2_VoxCPM2_nativeSmokeComponents(
    JNIEnv* env,
    jclass,
    jlong handle) {
    jni_guard(env, [&]() {
        with_synthesizer(handle, [](const voxcpm2::Synthesizer& synthesizer) {
            synthesizer.smoke_components();
        });
    });
}

extern "C" JNIEXPORT jint JNICALL Java_top_yurin_voxcpm2_VoxCPM2_nativeGetInputSampleRate(
    JNIEnv* env,
    jclass,
    jlong handle) {
    return jni_guard(env, jint{0}, [&]() {
        return static_cast<jint>(with_synthesizer(
            handle,
            [](const voxcpm2::Synthesizer& synthesizer) { return synthesizer.input_sample_rate(); }));
    });
}

extern "C" JNIEXPORT jint JNICALL Java_top_yurin_voxcpm2_VoxCPM2_nativeGetOutputSampleRate(
    JNIEnv* env,
    jclass,
    jlong handle) {
    return jni_guard(env, jint{0}, [&]() {
        return static_cast<jint>(with_synthesizer(
            handle,
            [](const voxcpm2::Synthesizer& synthesizer) { return synthesizer.output_sample_rate(); }));
    });
}

extern "C" JNIEXPORT jobjectArray JNICALL Java_top_yurin_voxcpm2_VoxCPM2_nativeGetMissingRequiredComponents(
    JNIEnv* env,
    jclass,
    jlong handle) {
    return jni_guard(env, static_cast<jobjectArray>(nullptr), [&]() {
        return java_strings(
            env,
            with_synthesizer(
                handle,
                [](const voxcpm2::Synthesizer& synthesizer) {
                    return synthesizer.missing_required_components();
                }));
    });
}
