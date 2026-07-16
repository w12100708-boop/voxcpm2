// Compile-time requirement for the C++26 resource inclusion directive.

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#if NCNN_VULKAN && !defined(__has_embed)
#error "VoxCPM2 Vulkan layers require a C++ compiler with #embed support"
#endif
