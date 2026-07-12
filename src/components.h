// Internal VoxCPM2 component metadata

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <string_view>

namespace voxcpm2::runtime {

enum class Component : std::size_t {
    text_embed,
    base_decoder_kv,
    residual_decoder_kv,
    fsq,
    fusion_proj,
    dit_proj,
    stop_head,
    feat_encoder,
    dit_estimator,
    audio_vae_encoder,
    audio_vae_decoder,
    count,
};

struct ComponentSpec {
    Component id;
    std::string_view name;
    bool dual_backend = false;
};

inline constexpr std::array<ComponentSpec, static_cast<std::size_t>(Component::count)> kComponentSpecs = {{
    {Component::text_embed, "text_embed", false},
    {Component::base_decoder_kv, "base_decoder_kv", true},
    {Component::residual_decoder_kv, "residual_decoder_kv", true},
    {Component::fsq, "fsq", false},
    {Component::fusion_proj, "fusion_proj", false},
    {Component::dit_proj, "dit_proj", false},
    {Component::stop_head, "stop_head", false},
    {Component::feat_encoder, "feat_encoder", true},
    {Component::dit_estimator, "dit_estimator", true},
    {Component::audio_vae_encoder, "audio_vae_encoder", false},
    {Component::audio_vae_decoder, "audio_vae_decoder", false},
}};

constexpr const ComponentSpec& component_spec(Component component) {
    return kComponentSpecs[static_cast<std::size_t>(component)];
}

inline std::string resolve_param_key(Component component, bool use_vulkan) {
    const ComponentSpec& spec = component_spec(component);
    if (spec.dual_backend) {
        return std::string(spec.name) + (use_vulkan ? ".vulkan" : ".cpu");
    }
    return std::string(spec.name);
}

} // namespace voxcpm2::runtime
