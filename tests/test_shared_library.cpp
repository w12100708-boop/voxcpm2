// Shared-library public API smoke test

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "voxcpm2/tokenizer.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

int main() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "voxcpm2_shared_library_tokenizer.json";
    {
        std::ofstream output(path);
        output << R"({"model":{"vocab":{"▁":0,"a":1},"merges":[]}})";
        if (not output) {
            throw std::runtime_error("failed to write shared-library tokenizer fixture");
        }
    }

    const voxcpm2::Tokenizer tokenizer = voxcpm2::Tokenizer::from_file(path);
    std::filesystem::remove(path);
    if (tokenizer.encode("a") != std::vector<int>({0, 1})) {
        throw std::runtime_error("shared-library tokenizer result mismatch");
    }
    return 0;
}
