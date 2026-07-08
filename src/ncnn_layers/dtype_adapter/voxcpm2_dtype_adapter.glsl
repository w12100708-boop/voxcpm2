// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#version 450

layout(binding = 0) readonly buffer bottom_blob { sfp bottom_blob_data[]; };
layout(binding = 1) writeonly buffer top_blob { float top_blob_data[]; };

layout(constant_id = 0) const uint n = 0;

layout(push_constant) uniform parameter
{
    uint n;
} p;

void main()
{
    const uint gx = gl_GlobalInvocationID.x;
    if (gx >= psc(n))
        return;

    top_blob_data[gx] = float(buffer_ld1(bottom_blob_data, gx));
}
