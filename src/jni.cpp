// JNI adapter for the stable VoxCPM2 C ABI

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "voxcpm2/c_api.h"

#include <jni.h>

#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

class PendingJavaException final {};

class StringChars final {
public:
    StringChars(JNIEnv* env, jstring value)
        : env_(env), value_(value), chars_(env->GetStringChars(value, nullptr)) {
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

class FloatArrayView final {
public:
    FloatArrayView(JNIEnv* env, jfloatArray value) : env_(env), value_(value) {
        if (value_ == nullptr) {
            return;
        }
        length_ = env_->GetArrayLength(value_);
        values_ = env_->GetFloatArrayElements(value_, nullptr);
        if (values_ == nullptr) {
            throw PendingJavaException{};
        }
    }

    ~FloatArrayView() {
        if (values_ != nullptr) {
            env_->ReleaseFloatArrayElements(value_, values_, JNI_ABORT);
        }
    }

    FloatArrayView(const FloatArrayView&) = delete;
    FloatArrayView& operator=(const FloatArrayView&) = delete;

    [[nodiscard]] bool present() const {
        return value_ != nullptr;
    }

    [[nodiscard]] const float* data() const {
        return values_;
    }

    [[nodiscard]] std::size_t size() const {
        return static_cast<std::size_t>(length_);
    }

private:
    JNIEnv* env_;
    jfloatArray value_ = nullptr;
    jfloat* values_ = nullptr;
    jsize length_ = 0;
};

void append_utf8(std::string& output, std::uint32_t codepoint) {
    if (codepoint == 0) {
        throw std::invalid_argument("strings passed to VoxCPM2 must not contain NUL");
    }
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

void throw_java(JNIEnv* env, const char* class_name, std::string_view message) noexcept {
    if (env->ExceptionCheck()) {
        return;
    }
    jclass exception_class = env->FindClass(class_name);
    if (exception_class == nullptr) {
        return;
    }
    const std::string stable_message(message);
    env->ThrowNew(exception_class, stable_message.c_str());
    env->DeleteLocalRef(exception_class);
}

bool accept_status(JNIEnv* env, voxcpm2_status status, voxcpm2_error& error) noexcept {
    if (status == VOXCPM2_STATUS_OK) {
        voxcpm2_error_free(&error);
        return true;
    }
    const std::string_view message = error.message == nullptr ? "VoxCPM2 native operation failed" : error.message;
    switch (status) {
    case VOXCPM2_STATUS_INVALID_ARGUMENT:
        throw_java(env, "java/lang/IllegalArgumentException", message);
        break;
    case VOXCPM2_STATUS_INVALID_STATE:
        throw_java(env, "java/lang/IllegalStateException", message);
        break;
    case VOXCPM2_STATUS_CANCELLED:
        throw_java(env, "java/util/concurrent/CancellationException", message);
        break;
    case VOXCPM2_STATUS_OUT_OF_MEMORY:
        throw_java(env, "java/lang/OutOfMemoryError", message);
        break;
    case VOXCPM2_STATUS_RUNTIME_ERROR:
        throw_java(env, "top/yurin/voxcpm2/VoxCPM2Exception", message);
        break;
    case VOXCPM2_STATUS_OK:
        break;
    }
    voxcpm2_error_free(&error);
    return false;
}

template <typename Result, typename Callable>
Result jni_guard(JNIEnv* env, Result fallback, Callable&& callable) noexcept {
    try {
        return std::forward<Callable>(callable)();
    } catch (const PendingJavaException&) {
        return fallback;
    } catch (const std::invalid_argument& error) {
        throw_java(env, "java/lang/IllegalArgumentException", error.what());
    } catch (const std::bad_alloc&) {
        throw_java(env, "java/lang/OutOfMemoryError", "native allocation failed");
    } catch (const std::exception& error) {
        throw_java(env, "top/yurin/voxcpm2/VoxCPM2Exception", error.what());
    } catch (...) {
        throw_java(env, "top/yurin/voxcpm2/VoxCPM2Exception", "unknown JNI adapter exception");
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

template <typename Type>
Type* native_pointer(jlong value) {
    return reinterpret_cast<Type*>(static_cast<std::uintptr_t>(value));
}

template <typename Type>
jlong native_handle(Type* value) {
    return static_cast<jlong>(reinterpret_cast<std::uintptr_t>(value));
}

struct JniProgressContext {
    JNIEnv* env = nullptr;
    jobject callback = nullptr;
    jmethodID method = nullptr;
};

JniProgressContext progress_context(JNIEnv* env, jobject callback) {
    JniProgressContext context{.env = env, .callback = callback, .method = nullptr};
    if (callback == nullptr) {
        return context;
    }
    jclass callback_class = env->GetObjectClass(callback);
    if (callback_class == nullptr) {
        throw PendingJavaException{};
    }
    context.method = env->GetMethodID(callback_class, "onProgress", "(ILjava/lang/String;II)Z");
    env->DeleteLocalRef(callback_class);
    if (context.method == nullptr) {
        throw PendingJavaException{};
    }
    return context;
}

int32_t report_progress(
    void* user_data,
    voxcpm2_progress_phase phase,
    const char* label,
    int32_t completed,
    int32_t total) noexcept {
    auto* context = static_cast<JniProgressContext*>(user_data);
    if (context == nullptr or context->callback == nullptr) {
        return 1;
    }
    jstring java_label = context->env->NewStringUTF(label == nullptr ? "" : label);
    if (java_label == nullptr) {
        return 0;
    }
    const jboolean keep_going = context->env->CallBooleanMethod(
        context->callback,
        context->method,
        static_cast<jint>(phase),
        java_label,
        static_cast<jint>(completed),
        static_cast<jint>(total));
    context->env->DeleteLocalRef(java_label);
    return context->env->ExceptionCheck() or keep_going != JNI_TRUE ? 0 : 1;
}

voxcpm2_progress_callback progress_callback(const JniProgressContext& context) {
    return context.callback == nullptr ? nullptr : report_progress;
}

jint native_abi_version(JNIEnv*, jclass) {
    return static_cast<jint>(voxcpm2_c_abi_version());
}

jlong native_create_operation(JNIEnv* env, jclass) {
    voxcpm2_operation* operation = voxcpm2_operation_create();
    if (operation == nullptr) {
        throw_java(env, "java/lang/OutOfMemoryError", "native allocation failed");
        return 0;
    }
    return native_handle(operation);
}

void native_cancel_operation(JNIEnv*, jclass, jlong operation) {
    voxcpm2_operation_cancel(native_pointer<voxcpm2_operation>(operation));
}

void native_destroy_operation(JNIEnv*, jclass, jlong operation) {
    voxcpm2_operation_destroy(native_pointer<voxcpm2_operation>(operation));
}

jlong native_create(
    JNIEnv* env,
    jclass,
    jstring model_dir,
    jboolean use_vulkan,
    jboolean profile,
    jint threads,
    jint vulkan_device,
    jlong operation,
    jobject callback) {
    return jni_guard(env, jlong{0}, [&]() {
        const std::string model_path = java_string(env, model_dir, "modelDirectory");
        JniProgressContext context = progress_context(env, callback);
        voxcpm2_synthesizer* synthesizer = nullptr;
        voxcpm2_error error{};
        const voxcpm2_config config{
            .model_dir_utf8 = model_path.c_str(),
            .use_vulkan = use_vulkan == JNI_TRUE ? 1 : 0,
            .profile = profile == JNI_TRUE ? 1 : 0,
            .threads = static_cast<int32_t>(threads),
            .vulkan_device = static_cast<int32_t>(vulkan_device),
        };
        const voxcpm2_status status = voxcpm2_synthesizer_create(
            &config,
            native_pointer<voxcpm2_operation>(operation),
            progress_callback(context),
            &context,
            &synthesizer,
            &error);
        if (env->ExceptionCheck()) {
            voxcpm2_error_free(&error);
            voxcpm2_synthesizer_destroy(synthesizer);
            return jlong{0};
        }
        if (not accept_status(env, status, error)) {
            return jlong{0};
        }
        return native_handle(synthesizer);
    });
}

void native_destroy(JNIEnv*, jclass, jlong synthesizer) {
    voxcpm2_synthesizer_destroy(native_pointer<voxcpm2_synthesizer>(synthesizer));
}

jint input_sample_rate(JNIEnv* env, voxcpm2_synthesizer* synthesizer) {
    int32_t sample_rate = 0;
    voxcpm2_error error{};
    const voxcpm2_status status = voxcpm2_synthesizer_input_sample_rate(synthesizer, &sample_rate, &error);
    return accept_status(env, status, error) ? static_cast<jint>(sample_rate) : 0;
}

jfloatArray native_generate(
    JNIEnv* env,
    jclass,
    jlong synthesizer_handle,
    jstring text,
    jstring prompt_text,
    jfloatArray prompt_audio,
    jfloatArray reference_audio,
    jint min_patches,
    jint inference_timesteps,
    jfloat cfg_value,
    jlong operation,
    jobject callback) {
    return jni_guard(env, static_cast<jfloatArray>(nullptr), [&]() {
        auto* synthesizer = native_pointer<voxcpm2_synthesizer>(synthesizer_handle);
        const std::string cpp_text = java_string(env, text, "text");
        const std::string cpp_prompt_text = java_string(env, prompt_text, "promptText", true);
        const jint sample_rate = input_sample_rate(env, synthesizer);
        if (env->ExceptionCheck()) {
            return static_cast<jfloatArray>(nullptr);
        }
        FloatArrayView prompt_view(env, prompt_audio);
        FloatArrayView reference_view(env, reference_audio);
        const voxcpm2_audio_view c_prompt{
            .samples = prompt_view.data(),
            .sample_count = prompt_view.size(),
            .sample_rate = sample_rate,
            .channels = 1,
        };
        const voxcpm2_audio_view c_reference{
            .samples = reference_view.data(),
            .sample_count = reference_view.size(),
            .sample_rate = sample_rate,
            .channels = 1,
        };
        const voxcpm2_synthesis_options options{
            .text_utf8 = cpp_text.c_str(),
            .prompt_text_utf8 = prompt_text == nullptr ? nullptr : cpp_prompt_text.c_str(),
            .prompt_audio = prompt_view.present() ? &c_prompt : nullptr,
            .reference_audio = reference_view.present() ? &c_reference : nullptr,
            .min_patches = static_cast<int32_t>(min_patches),
            .inference_timesteps = static_cast<int32_t>(inference_timesteps),
            .cfg_value = static_cast<float>(cfg_value),
        };
        JniProgressContext context = progress_context(env, callback);
        voxcpm2_audio_buffer audio{};
        voxcpm2_error error{};
        const voxcpm2_status status = voxcpm2_synthesizer_generate(
            synthesizer,
            &options,
            native_pointer<voxcpm2_operation>(operation),
            progress_callback(context),
            &context,
            &audio,
            &error);
        if (env->ExceptionCheck()) {
            voxcpm2_error_free(&error);
            voxcpm2_audio_buffer_free(&audio);
            return static_cast<jfloatArray>(nullptr);
        }
        if (not accept_status(env, status, error)) {
            voxcpm2_audio_buffer_free(&audio);
            return static_cast<jfloatArray>(nullptr);
        }
        if (audio.sample_count > static_cast<std::size_t>(std::numeric_limits<jsize>::max())) {
            voxcpm2_audio_buffer_free(&audio);
            throw std::length_error("generated audio is too large for a Java float array");
        }
        const jsize length = static_cast<jsize>(audio.sample_count);
        jfloatArray result = env->NewFloatArray(length);
        if (result == nullptr) {
            voxcpm2_audio_buffer_free(&audio);
            throw PendingJavaException{};
        }
        if (length > 0) {
            env->SetFloatArrayRegion(result, 0, length, audio.samples);
        }
        voxcpm2_audio_buffer_free(&audio);
        if (env->ExceptionCheck()) {
            env->DeleteLocalRef(result);
            throw PendingJavaException{};
        }
        return result;
    });
}

void native_smoke_components(
    JNIEnv* env,
    jclass,
    jlong synthesizer,
    jlong operation,
    jobject callback) {
    jni_guard(env, [&]() {
        JniProgressContext context = progress_context(env, callback);
        voxcpm2_error error{};
        const voxcpm2_status status = voxcpm2_synthesizer_smoke_components(
            native_pointer<voxcpm2_synthesizer>(synthesizer),
            native_pointer<voxcpm2_operation>(operation),
            progress_callback(context),
            &context,
            &error);
        if (env->ExceptionCheck()) {
            voxcpm2_error_free(&error);
            return;
        }
        static_cast<void>(accept_status(env, status, error));
    });
}

jint native_get_input_sample_rate(JNIEnv* env, jclass, jlong synthesizer) {
    return input_sample_rate(env, native_pointer<voxcpm2_synthesizer>(synthesizer));
}

jint native_get_output_sample_rate(JNIEnv* env, jclass, jlong synthesizer) {
    int32_t sample_rate = 0;
    voxcpm2_error error{};
    const voxcpm2_status status = voxcpm2_synthesizer_output_sample_rate(
        native_pointer<voxcpm2_synthesizer>(synthesizer),
        &sample_rate,
        &error);
    return accept_status(env, status, error) ? static_cast<jint>(sample_rate) : 0;
}

jobjectArray native_get_missing_required_components(JNIEnv* env, jclass, jlong synthesizer) {
    return jni_guard(env, static_cast<jobjectArray>(nullptr), [&]() {
        voxcpm2_string_list components{};
        voxcpm2_error error{};
        const voxcpm2_status status = voxcpm2_synthesizer_missing_required_components(
            native_pointer<voxcpm2_synthesizer>(synthesizer),
            &components,
            &error);
        if (not accept_status(env, status, error)) {
            return static_cast<jobjectArray>(nullptr);
        }
        if (components.count > static_cast<std::size_t>(std::numeric_limits<jsize>::max())) {
            voxcpm2_string_list_free(&components);
            throw std::length_error("too many missing component names for a Java array");
        }
        jclass string_class = env->FindClass("java/lang/String");
        if (string_class == nullptr) {
            voxcpm2_string_list_free(&components);
            throw PendingJavaException{};
        }
        jobjectArray result = env->NewObjectArray(static_cast<jsize>(components.count), string_class, nullptr);
        env->DeleteLocalRef(string_class);
        if (result == nullptr) {
            voxcpm2_string_list_free(&components);
            throw PendingJavaException{};
        }
        for (std::size_t index = 0; index < components.count; ++index) {
            jstring item = env->NewStringUTF(components.items[index]);
            if (item == nullptr) {
                env->DeleteLocalRef(result);
                voxcpm2_string_list_free(&components);
                throw PendingJavaException{};
            }
            env->SetObjectArrayElement(result, static_cast<jsize>(index), item);
            env->DeleteLocalRef(item);
            if (env->ExceptionCheck()) {
                env->DeleteLocalRef(result);
                voxcpm2_string_list_free(&components);
                throw PendingJavaException{};
            }
        }
        voxcpm2_string_list_free(&components);
        return result;
    });
}

const JNINativeMethod kMethods[] = {
    {const_cast<char*>("nativeAbiVersion"), const_cast<char*>("()I"), reinterpret_cast<void*>(native_abi_version)},
    {const_cast<char*>("nativeCreateOperation"), const_cast<char*>("()J"), reinterpret_cast<void*>(native_create_operation)},
    {const_cast<char*>("nativeCancelOperation"), const_cast<char*>("(J)V"), reinterpret_cast<void*>(native_cancel_operation)},
    {const_cast<char*>("nativeDestroyOperation"), const_cast<char*>("(J)V"), reinterpret_cast<void*>(native_destroy_operation)},
    {const_cast<char*>("nativeCreate"), const_cast<char*>("(Ljava/lang/String;ZZIIJLtop/yurin/voxcpm2/internal/JniProgressCallback;)J"), reinterpret_cast<void*>(native_create)},
    {const_cast<char*>("nativeDestroy"), const_cast<char*>("(J)V"), reinterpret_cast<void*>(native_destroy)},
    {const_cast<char*>("nativeGenerate"), const_cast<char*>("(JLjava/lang/String;Ljava/lang/String;[F[FIIFJLtop/yurin/voxcpm2/internal/JniProgressCallback;)[F"), reinterpret_cast<void*>(native_generate)},
    {const_cast<char*>("nativeSmokeComponents"), const_cast<char*>("(JJLtop/yurin/voxcpm2/internal/JniProgressCallback;)V"), reinterpret_cast<void*>(native_smoke_components)},
    {const_cast<char*>("nativeGetInputSampleRate"), const_cast<char*>("(J)I"), reinterpret_cast<void*>(native_get_input_sample_rate)},
    {const_cast<char*>("nativeGetOutputSampleRate"), const_cast<char*>("(J)I"), reinterpret_cast<void*>(native_get_output_sample_rate)},
    {const_cast<char*>("nativeGetMissingRequiredComponents"), const_cast<char*>("(J)[Ljava/lang/String;"), reinterpret_cast<void*>(native_get_missing_required_components)},
};

} // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK or env == nullptr) {
        return JNI_ERR;
    }
    jclass bindings = env->FindClass("top/yurin/voxcpm2/internal/JniBindings");
    if (bindings == nullptr) {
        return JNI_ERR;
    }
    const jint result = env->RegisterNatives(
        bindings,
        kMethods,
        static_cast<jint>(sizeof(kMethods) / sizeof(kMethods[0])));
    env->DeleteLocalRef(bindings);
    return result == JNI_OK ? JNI_VERSION_1_6 : JNI_ERR;
}
