// Stable C ABI for the VoxCPM2 synthesis runtime

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(VOXCPM2_SHARED_BUILD)
#define VOXCPM2_C_API __declspec(dllexport)
#elif defined(VOXCPM2_SHARED_USE)
#define VOXCPM2_C_API __declspec(dllimport)
#else
#define VOXCPM2_C_API
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define VOXCPM2_C_API __attribute__((visibility("default")))
#else
#define VOXCPM2_C_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define VOXCPM2_C_ABI_VERSION_MAJOR 1u
#define VOXCPM2_C_ABI_VERSION_MINOR 0u
#define VOXCPM2_C_ABI_VERSION ((VOXCPM2_C_ABI_VERSION_MAJOR << 16u) | VOXCPM2_C_ABI_VERSION_MINOR)

typedef struct voxcpm2_synthesizer voxcpm2_synthesizer;
typedef struct voxcpm2_operation voxcpm2_operation;

typedef enum voxcpm2_status {
    VOXCPM2_STATUS_OK = 0,
    VOXCPM2_STATUS_INVALID_ARGUMENT = 1,
    VOXCPM2_STATUS_INVALID_STATE = 2,
    VOXCPM2_STATUS_CANCELLED = 3,
    VOXCPM2_STATUS_OUT_OF_MEMORY = 4,
    VOXCPM2_STATUS_RUNTIME_ERROR = 5,
} voxcpm2_status;

typedef enum voxcpm2_progress_phase {
    VOXCPM2_PROGRESS_MODEL_LOAD = 0,
    VOXCPM2_PROGRESS_PREFIX = 1,
    VOXCPM2_PROGRESS_GENERATION = 2,
    VOXCPM2_PROGRESS_DECODE = 3,
    VOXCPM2_PROGRESS_SMOKE = 4,
} voxcpm2_progress_phase;

typedef struct voxcpm2_config {
    const char* model_dir_utf8;
    int32_t use_vulkan;
    int32_t profile;
    int32_t threads;
    int32_t vulkan_device;
} voxcpm2_config;

typedef struct voxcpm2_audio_view {
    const float* samples;
    size_t sample_count;
    int32_t sample_rate;
    int32_t channels;
} voxcpm2_audio_view;

typedef struct voxcpm2_synthesis_options {
    const char* text_utf8;
    const char* prompt_text_utf8;
    const voxcpm2_audio_view* prompt_audio;
    const voxcpm2_audio_view* reference_audio;
    int32_t min_patches;
    int32_t inference_timesteps;
    float cfg_value;
} voxcpm2_synthesis_options;

typedef struct voxcpm2_audio_buffer {
    float* samples;
    size_t sample_count;
    int32_t sample_rate;
    int32_t channels;
} voxcpm2_audio_buffer;

typedef struct voxcpm2_string_list {
    char** items;
    size_t count;
} voxcpm2_string_list;

typedef struct voxcpm2_error {
    voxcpm2_status code;
    char* message;
} voxcpm2_error;

// Functions returning voxcpm2_status initialize out_error when it is non-NULL.
// Release it with voxcpm2_error_free after every call, including success.

// Return non-zero to continue or zero to cancel the operation.
typedef int32_t (*voxcpm2_progress_callback)(
    void* user_data,
    voxcpm2_progress_phase phase,
    const char* label_utf8,
    int32_t completed,
    int32_t total);

VOXCPM2_C_API uint32_t voxcpm2_c_abi_version(void);

// An operation token is owned by the caller. Cancellation is thread-safe. The
// token must not be destroyed until the native call using it has returned.
VOXCPM2_C_API voxcpm2_operation* voxcpm2_operation_create(void);
VOXCPM2_C_API void voxcpm2_operation_cancel(voxcpm2_operation* operation);
VOXCPM2_C_API int32_t voxcpm2_operation_is_cancelled(const voxcpm2_operation* operation);
VOXCPM2_C_API void voxcpm2_operation_destroy(voxcpm2_operation* operation);

VOXCPM2_C_API voxcpm2_status voxcpm2_synthesizer_create(
    const voxcpm2_config* config,
    voxcpm2_operation* operation,
    voxcpm2_progress_callback progress,
    void* progress_user_data,
    voxcpm2_synthesizer** out_synthesizer,
    voxcpm2_error* out_error);

// Calls on one synthesizer are serialized. Destroy cancels an active operation,
// waits for active and queued callers to leave native code, and then releases
// the handle. Passing NULL is a no-op. No new call may begin once destroy has
// been invoked.
VOXCPM2_C_API void voxcpm2_synthesizer_destroy(voxcpm2_synthesizer* synthesizer);

VOXCPM2_C_API voxcpm2_status voxcpm2_synthesizer_generate(
    voxcpm2_synthesizer* synthesizer,
    const voxcpm2_synthesis_options* options,
    voxcpm2_operation* operation,
    voxcpm2_progress_callback progress,
    void* progress_user_data,
    voxcpm2_audio_buffer* out_audio,
    voxcpm2_error* out_error);

VOXCPM2_C_API voxcpm2_status voxcpm2_synthesizer_smoke_components(
    voxcpm2_synthesizer* synthesizer,
    voxcpm2_operation* operation,
    voxcpm2_progress_callback progress,
    void* progress_user_data,
    voxcpm2_error* out_error);

VOXCPM2_C_API voxcpm2_status voxcpm2_synthesizer_input_sample_rate(
    voxcpm2_synthesizer* synthesizer,
    int32_t* out_sample_rate,
    voxcpm2_error* out_error);

VOXCPM2_C_API voxcpm2_status voxcpm2_synthesizer_output_sample_rate(
    voxcpm2_synthesizer* synthesizer,
    int32_t* out_sample_rate,
    voxcpm2_error* out_error);

VOXCPM2_C_API voxcpm2_status voxcpm2_synthesizer_missing_required_components(
    voxcpm2_synthesizer* synthesizer,
    voxcpm2_string_list* out_components,
    voxcpm2_error* out_error);

// Successful output values are owned by the caller and must be released with
// the matching function. Each free function accepts NULL and clears its value.
VOXCPM2_C_API void voxcpm2_audio_buffer_free(voxcpm2_audio_buffer* buffer);
VOXCPM2_C_API void voxcpm2_string_list_free(voxcpm2_string_list* list);
VOXCPM2_C_API void voxcpm2_error_free(voxcpm2_error* error);

#ifdef __cplusplus
}
#endif
