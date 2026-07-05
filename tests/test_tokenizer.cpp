// VoxCPM2 tokenizer regression tests

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "voxcpm2/tokenizer.h"

#include <filesystem>
#include <fstream>
#include <print>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (not condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main() {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "voxcpm2_tokenizer_test.json";
    {
        std::ofstream out(path);
        out << R"({
  "model": {
    "vocab": {
      "▁": 0,
      "你": 1,
      "好": 2,
      "▁A": 3,
      "<0x21>": 4
    },
    "merges": [
      "▁ A"
    ]
  }
})";
    }

    const voxcpm2::Tokenizer tokenizer = voxcpm2::Tokenizer::from_file(path);
    require(tokenizer.encode("你好") == std::vector<int>({0, 1, 2}), "Chinese token fallback should preserve sentencepiece marker and split known chars");
    require(tokenizer.encode("A!") == std::vector<int>({3, 4}), "BPE merge and byte fallback should work");

    std::filesystem::remove(path);
    std::println("tokenizer tests passed");
    return 0;
}
