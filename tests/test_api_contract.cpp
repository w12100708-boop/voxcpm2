// Public API compile-time contract checks

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "voxcpm2/audio.h"
#include "voxcpm2/audio_io.h"
#include "voxcpm2/paged_kv_cache.h"
#include "voxcpm2/synthesizer.h"
#include "voxcpm2/tokenizer.h"

#include <filesystem>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

static_assert(std::is_same_v<decltype(voxcpm2::AudioBuffer::sample_rate), int>);
static_assert(std::is_same_v<decltype(voxcpm2::AudioBuffer::channels), int>);
static_assert(std::is_same_v<decltype(voxcpm2::AudioBuffer::samples), std::vector<float>>);

static_assert(std::is_constructible_v<voxcpm2::PagedKvCache, int>);
static_assert(std::is_constructible_v<voxcpm2::PagedKvCache, int, int>);
static_assert(std::is_same_v<decltype(std::declval<const voxcpm2::PagedKvCache&>().layer_count()), int>);
static_assert(std::is_same_v<decltype(std::declval<const voxcpm2::PagedKvCache&>().materialize(0)), voxcpm2::KvCachePair>);

static_assert(std::is_same_v<
              decltype(std::declval<const voxcpm2::Tokenizer&>().encode(std::declval<const std::string&>())),
              std::vector<int>>);
static_assert(std::is_same_v<
              decltype(voxcpm2::Tokenizer::from_file(std::declval<const std::filesystem::path&>())),
              voxcpm2::Tokenizer>);

static_assert(std::is_same_v<decltype(voxcpm2::SynthesisOptions::prompt_audio), std::optional<voxcpm2::AudioBuffer>>);
static_assert(std::is_same_v<decltype(voxcpm2::SynthesizerConfig::model_dir), std::filesystem::path>);
static_assert(not std::is_copy_constructible_v<voxcpm2::Synthesizer>);
static_assert(not std::is_copy_assignable_v<voxcpm2::Synthesizer>);
static_assert(std::is_move_constructible_v<voxcpm2::Synthesizer>);
static_assert(std::is_move_assignable_v<voxcpm2::Synthesizer>);

static_assert(std::is_same_v<
              decltype(voxcpm2::read_audio_file(std::declval<const std::filesystem::path&>(), 16000)),
              voxcpm2::AudioBuffer>);
static_assert(std::is_same_v<
              decltype(voxcpm2::write_audio_file(
                  std::declval<const std::filesystem::path&>(),
                  std::declval<const voxcpm2::AudioBuffer&>())),
              void>);

int main() {
    return 0;
}
