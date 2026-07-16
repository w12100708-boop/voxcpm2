// Verify that the public VoxCPM2 C ABI header is valid C11.

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "voxcpm2/c_api.h"

#include <stddef.h>
#include <stdint.h>

static int32_t progress_callback(
    void* user_data,
    voxcpm2_progress_phase phase,
    const char* label,
    int32_t completed,
    int32_t total) {
    (void)user_data;
    (void)phase;
    (void)label;
    (void)completed;
    (void)total;
    return 1;
}

int main(void) {
    voxcpm2_config config = {0};
    voxcpm2_audio_view view = {0};
    voxcpm2_synthesis_options options = {0};
    voxcpm2_audio_buffer output = {0};
    voxcpm2_string_list strings = {0};
    voxcpm2_error error = {0};
    voxcpm2_progress_callback callback = progress_callback;
    (void)config;
    (void)view;
    (void)options;
    (void)output;
    (void)strings;
    (void)error;
    (void)callback;
    return VOXCPM2_C_ABI_VERSION_MAJOR == 1u ? 0 : 1;
}
