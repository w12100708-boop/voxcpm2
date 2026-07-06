// Internal progress hook tests

// Copyright (c) 2026 Yurin <liyulin.china@gmail.com>
// Licensed under MIT. Not all rights reserved.
// SPDX-License-Identifier: MIT

#include "progress.h"

#include <stdexcept>
#include <string>

namespace {

class CountingSink final : public voxcpm2::progress::Sink {
public:
    void begin_phase(voxcpm2::progress::Phase, std::string_view label, int steps) override {
        last_label = std::string(label);
        last_steps = steps;
        ++begins;
    }

    void advance_phase(std::string_view label, int increment) override {
        last_label = std::string(label);
        advances += increment;
    }

    void current(std::string_view label, int current, int total) override {
        last_label = std::string(label);
        last_current = current;
        last_total = total;
        ++currents;
    }

    void finish_phase(std::string_view label) override {
        last_label = std::string(label);
        ++finishes;
    }

    int begins = 0;
    int advances = 0;
    int currents = 0;
    int finishes = 0;
    int last_steps = 0;
    int last_current = 0;
    int last_total = 0;
    std::string last_label;
};

void require(bool condition, const char* message) {
    if (not condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main() {
    require(voxcpm2::progress::current_sink() == nullptr, "progress sink should start empty");

    CountingSink outer;
    {
        voxcpm2::progress::SinkScope outer_scope(&outer);
        require(voxcpm2::progress::current_sink() == &outer, "outer sink not installed");
        voxcpm2::progress::begin_phase(voxcpm2::progress::Phase::generation, "generation", 7);
        voxcpm2::progress::current("dit_estimator", 3, 10);
        voxcpm2::progress::advance_phase("patch", 2);
        voxcpm2::progress::finish_phase("generation");

        require(outer.begins == 1, "begin event missing");
        require(outer.currents == 1, "current event missing");
        require(outer.advances == 2, "advance event missing");
        require(outer.finishes == 1, "finish event missing");
        require(outer.last_steps == 7, "begin steps not forwarded");
        require(outer.last_current == 3 and outer.last_total == 10, "current values not forwarded");

        CountingSink inner;
        {
            voxcpm2::progress::SinkScope inner_scope(&inner);
            require(voxcpm2::progress::current_sink() == &inner, "inner sink not installed");
        }
        require(voxcpm2::progress::current_sink() == &outer, "outer sink not restored");
    }

    require(voxcpm2::progress::current_sink() == nullptr, "progress sink should restore to empty");
    return 0;
}
