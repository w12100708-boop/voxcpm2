// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#version 450

layout(constant_id = 0) const int half_dim = 512;
layout(constant_id = 1) const float timestep_scale = 1000.0;
layout(constant_id = 2) const float log_max_period = 9.210340371976184;
layout(constant_id = 3) const int schedule_mode = 0;

layout(binding = 0) readonly buffer bottom_blob { sfp bottom_blob_data[]; };
layout(binding = 1) writeonly buffer top_blob { sfp top_blob_data[]; };

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    const uint gy = gl_GlobalInvocationID.y;
    const uint out_dim = uint(half_dim * 2);

    if (gx >= out_dim)
        return;

    float timestep = float(buffer_ld1(bottom_blob_data, gy));
    if (schedule_mode != 0)
    {
        const float step_index = float(buffer_ld1(bottom_blob_data, 0));
        const float timesteps = max(float(buffer_ld1(bottom_blob_data, 1)), 1.0);
        const float u = 1.0 - step_index / timesteps;
        timestep = u + (cos(1.57079632679489661923 * u) - 1.0 + u);
    }

    const float t = timestep * timestep_scale;
    const uint freq_i = gx < uint(half_dim) ? gx : gx - uint(half_dim);
    const float inv_freq = exp(-log_max_period * float(freq_i) / float(half_dim - 1));
    const float phase = t * inv_freq;
    const float value = gx < uint(half_dim) ? sin(phase) : cos(phase);
    buffer_st1(top_blob_data, gy * out_dim + gx, sfp(value));
}
