// VoxCPM2 tokenizer public API

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace voxcpm2 {

class Tokenizer {
public:
    static Tokenizer from_file(const std::filesystem::path& path);

    [[nodiscard]] std::vector<int> encode(const std::string& text) const;

private:
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::string, int> merges_rank_;
    std::unordered_set<std::string> multichar_chinese_tokens_;
};

} // namespace voxcpm2
