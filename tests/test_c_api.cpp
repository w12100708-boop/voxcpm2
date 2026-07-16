// Stable C ABI behavior tests

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "voxcpm2/c_api.h"

#include <cassert>
#include <cstdint>

int main() {
    assert(voxcpm2_c_abi_version() == VOXCPM2_C_ABI_VERSION);

    voxcpm2_operation* operation = voxcpm2_operation_create();
    assert(operation != nullptr);
    assert(voxcpm2_operation_is_cancelled(operation) == 0);
    voxcpm2_operation_cancel(operation);
    assert(voxcpm2_operation_is_cancelled(operation) == 1);

    const voxcpm2_config config{
        .model_dir_utf8 = "/path/not/consulted/after/pre-cancellation",
        .use_vulkan = 0,
        .profile = 0,
        .threads = 1,
        .vulkan_device = 0,
    };
    voxcpm2_synthesizer* synthesizer = nullptr;
    voxcpm2_error error{};
    assert(voxcpm2_synthesizer_create(
               &config,
               operation,
               nullptr,
               nullptr,
               &synthesizer,
               &error) == VOXCPM2_STATUS_CANCELLED);
    assert(synthesizer == nullptr);
    assert(error.code == VOXCPM2_STATUS_CANCELLED);
    assert(error.message != nullptr);
    voxcpm2_error_free(&error);
    assert(error.code == VOXCPM2_STATUS_OK);
    assert(error.message == nullptr);
    voxcpm2_operation_destroy(operation);

    assert(voxcpm2_synthesizer_create(
               nullptr,
               nullptr,
               nullptr,
               nullptr,
               &synthesizer,
               &error) == VOXCPM2_STATUS_INVALID_ARGUMENT);
    assert(synthesizer == nullptr);
    assert(error.code == VOXCPM2_STATUS_INVALID_ARGUMENT);
    voxcpm2_error_free(&error);

    voxcpm2_audio_buffer audio{};
    voxcpm2_audio_buffer_free(&audio);
    voxcpm2_string_list strings{};
    voxcpm2_string_list_free(&strings);
    voxcpm2_synthesizer_destroy(nullptr);
    voxcpm2_operation_cancel(nullptr);
    voxcpm2_operation_destroy(nullptr);
    return 0;
}
