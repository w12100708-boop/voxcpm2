// VoxCPM2 tokenizer implementation

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "voxcpm2/tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <format>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

namespace voxcpm2 {
namespace {

using json = nlohmann::json;

struct TokenizerModel {
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> merges;
};

struct TokenizerDocument {
    TokenizerModel model;
};

void from_json(const json& document, TokenizerModel& model) {
    document.at("vocab").get_to(model.vocab);
    document.at("merges").get_to(model.merges);
}

void from_json(const json& document, TokenizerDocument& tokenizer) {
    document.at("model").get_to(tokenizer.model);
}

std::string pair_key(const std::string& a, const std::string& b) {
    std::string key;
    key.reserve(a.size() + 1 + b.size());
    key.append(a);
    key.push_back('\t');
    key.append(b);
    return key;
}

bool next_utf8(const std::string& s, std::size_t& i, std::uint32_t& cp) {
    if (i >= s.size()) {
        return false;
    }
    const auto c0 = static_cast<unsigned char>(s[i]);
    if (c0 < 0x80) {
        cp = c0;
        ++i;
        return true;
    }
    if ((c0 >> 5) == 0x6) {
        if (i + 1 >= s.size()) {
            return false;
        }
        const auto c1 = static_cast<unsigned char>(s[i + 1]);
        if ((c1 & 0xc0) != 0x80) {
            return false;
        }
        cp = ((c0 & 0x1f) << 6) | (c1 & 0x3f);
        i += 2;
        return true;
    }
    if ((c0 >> 4) == 0xe) {
        if (i + 2 >= s.size()) {
            return false;
        }
        const auto c1 = static_cast<unsigned char>(s[i + 1]);
        const auto c2 = static_cast<unsigned char>(s[i + 2]);
        if ((c1 & 0xc0) != 0x80 or (c2 & 0xc0) != 0x80) {
            return false;
        }
        cp = ((c0 & 0x0f) << 12) | ((c1 & 0x3f) << 6) | (c2 & 0x3f);
        i += 3;
        return true;
    }
    if ((c0 >> 3) == 0x1e) {
        if (i + 3 >= s.size()) {
            return false;
        }
        const auto c1 = static_cast<unsigned char>(s[i + 1]);
        const auto c2 = static_cast<unsigned char>(s[i + 2]);
        const auto c3 = static_cast<unsigned char>(s[i + 3]);
        if ((c1 & 0xc0) != 0x80 or (c2 & 0xc0) != 0x80 or (c3 & 0xc0) != 0x80) {
            return false;
        }
        cp = ((c0 & 0x07) << 18) | ((c1 & 0x3f) << 12) | ((c2 & 0x3f) << 6) | (c3 & 0x3f);
        i += 4;
        return true;
    }
    return false;
}

bool is_chinese_codepoint(std::uint32_t cp) {
    return cp >= 0x4e00 and cp <= 0x9fff;
}

bool is_multichar_chinese_token(const std::string& token) {
    std::size_t i = 0;
    int count = 0;
    while (i < token.size()) {
        std::uint32_t cp = 0;
        if (not next_utf8(token, i, cp) or not is_chinese_codepoint(cp)) {
            return false;
        }
        ++count;
    }
    return count >= 2;
}

std::vector<std::string> utf8_chars(const std::string& s) {
    std::vector<std::string> chars;
    std::size_t i = 0;
    while (i < s.size()) {
        const std::size_t start = i;
        std::uint32_t cp = 0;
        if (not next_utf8(s, i, cp)) {
            break;
        }
        chars.emplace_back(s.substr(start, i - start));
    }
    return chars;
}

bool is_unicode_space(std::uint32_t cp) {
    if (cp <= 0x7f) {
        return cp == ' ' or cp == '\t' or cp == '\n' or cp == '\r' or cp == '\f' or cp == '\v';
    }
    switch (cp) {
    case 0x00a0:
    case 0x1680:
    case 0x2000:
    case 0x2001:
    case 0x2002:
    case 0x2003:
    case 0x2004:
    case 0x2005:
    case 0x2006:
    case 0x2007:
    case 0x2008:
    case 0x2009:
    case 0x200a:
    case 0x2028:
    case 0x2029:
    case 0x202f:
    case 0x205f:
    case 0x3000:
        return true;
    default:
        return false;
    }
}

std::vector<std::string> sentencepiece_pieces(const std::string& text) {
    static const std::string ws = "\xe2\x96\x81";
    std::vector<std::string> pieces;
    std::string current;
    std::size_t i = 0;
    while (i < text.size()) {
        const std::size_t start = i;
        std::uint32_t cp = 0;
        if (not next_utf8(text, i, cp)) {
            break;
        }
        if (is_unicode_space(cp)) {
            if (not current.empty()) {
                pieces.push_back(ws + current);
                current.clear();
            }
        } else {
            current.append(text, start, i - start);
        }
    }
    if (not current.empty()) {
        pieces.push_back(ws + current);
    }
    return pieces;
}

std::string remove_sentencepiece_marker(const std::string& token) {
    static const std::string marker = "\xe2\x96\x81";
    std::string out;
    out.reserve(token.size());
    std::size_t pos = 0;
    while (pos < token.size()) {
        if (token.compare(pos, marker.size(), marker) == 0) {
            pos += marker.size();
        } else {
            out.push_back(token[pos++]);
        }
    }
    return out;
}

void append_token_id_or_byte_fallback(const std::unordered_map<std::string, int>& token_to_id,
                                      const std::string& token,
                                      std::vector<int>& ids) {
    auto it = token_to_id.find(token);
    if (it != token_to_id.end()) {
        ids.push_back(it->second);
        return;
    }

    for (unsigned char byte : token) {
        const std::string byte_token = std::format("<0x{:02X}>", byte);
        it = token_to_id.find(byte_token);
        if (it == token_to_id.end()) [[unlikely]] {
            throw std::runtime_error("tokenizer produced unknown token: " + token);
        }
        ids.push_back(it->second);
    }
}

} // namespace

Tokenizer Tokenizer::from_file(const std::filesystem::path& path) {
    std::ifstream ifs(path);
    if (not ifs) [[unlikely]] {
        throw std::runtime_error("cannot open tokenizer: " + path.string());
    }

    TokenizerDocument document;
    try {
        json tokenizer_json;
        ifs >> tokenizer_json;
        document = tokenizer_json.get<TokenizerDocument>();
    } catch (const json::exception& error) {
        throw std::runtime_error(std::format("invalid tokenizer {}: {}", path.string(), error.what()));
    }

    Tokenizer out;
    out.token_to_id_ = std::move(document.model.vocab);

    for (const auto& [token, id] : out.token_to_id_) {
        (void)id;
        if (is_multichar_chinese_token(token)) {
            out.multichar_chinese_tokens_.insert(token);
        }
    }

    int rank = 0;
    for (const std::string& line : document.model.merges) {
        const std::size_t pos = line.find(' ');
        if (pos == std::string::npos) {
            continue;
        }
        out.merges_rank_[pair_key(line.substr(0, pos), line.substr(pos + 1))] = rank++;
    }
    return out;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    for (const auto& piece : sentencepiece_pieces(text)) {
        std::vector<std::string> symbols = utf8_chars(piece);
        while (symbols.size() >= 2) {
            int best_rank = std::numeric_limits<int>::max();
            int best_i = -1;
            for (int i = 0; i + 1 < static_cast<int>(symbols.size()); ++i) {
                auto it = merges_rank_.find(pair_key(symbols[i], symbols[i + 1]));
                if (it != merges_rank_.end() and it->second < best_rank) {
                    best_rank = it->second;
                    best_i = i;
                }
            }
            if (best_i < 0) {
                break;
            }
            symbols[best_i] += symbols[best_i + 1];
            symbols.erase(symbols.begin() + best_i + 1);
        }

        for (const auto& token : symbols) {
            const std::string clean_token = remove_sentencepiece_marker(token);
            if (multichar_chinese_tokens_.contains(clean_token)) {
                for (const auto& ch : utf8_chars(clean_token)) {
                    append_token_id_or_byte_fallback(token_to_id_, ch, ids);
                }
            } else {
                append_token_id_or_byte_fallback(token_to_id_, token, ids);
            }
        }
    }
    return ids;
}

} // namespace voxcpm2
