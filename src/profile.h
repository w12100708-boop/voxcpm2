// Internal runtime profiler for VoxCPM2 synthesis

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string_view>

#ifdef VOXCPM2_ENABLE_PROFILE

#include <chrono>
#include <string>
#include <vector>

namespace voxcpm2::runtime {

class Profile {
public:
    class Scope {
    public:
        Scope(Profile& profile, std::string_view label);
        ~Scope();

        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&& other) noexcept;
        Scope& operator=(Scope&&) = delete;

    private:
        using Clock = std::chrono::steady_clock;

        Profile* profile_ = nullptr;
        std::string label_;
        Clock::time_point start_;
    };

    explicit Profile(bool enabled);

    [[nodiscard]] Scope scope(std::string_view label);
    void count(std::string_view label, std::uint64_t increment = 1);
    void report(std::string_view total_label, double audio_seconds = 0.0) const;

    [[nodiscard]] bool enabled() const;

private:
    friend class Scope;
    using Clock = std::chrono::steady_clock;

    struct Timing {
        std::string label;
        std::uint64_t calls = 0;
        double total_ms = 0.0;
    };

    struct Counter {
        std::string label;
        std::uint64_t value = 0;
    };

    void record(std::string_view label, double elapsed_ms);

    bool enabled_ = false;
    Clock::time_point start_;
    std::vector<Timing> timings_;
    std::vector<Counter> counters_;
};

} // namespace voxcpm2::runtime

#else

namespace voxcpm2::runtime {

class Profile {
public:
    class Scope {};

    explicit constexpr Profile(bool) {}

    [[nodiscard]] constexpr Scope scope(std::string_view) const {
        return {};
    }

    constexpr void count(std::string_view, std::uint64_t = 1) const {}
    constexpr void report(std::string_view, double = 0.0) const {}

    [[nodiscard]] constexpr bool enabled() const {
        return false;
    }
};

} // namespace voxcpm2::runtime

#endif
