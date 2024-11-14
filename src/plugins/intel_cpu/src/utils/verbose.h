// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef CPU_DEBUG_CAPS

#include <node.h>

#include <string>
#include <cstdlib>
#include <sstream>

namespace ov {
namespace intel_cpu {

class Verbose {
public:
    Verbose(const NodePtr& _node, const std::string& _lvl)
        : node(_node),
          lvl(atoi(_lvl.c_str()) % 10),
          /* 1,  2,  3,  etc -> no color. 11, 22, 33, etc -> colorize */
          colorUp(atoi(_lvl.c_str()) / 10) {
        if (!shouldBePrinted())
            return;
        printInfo();
    }

    ~Verbose() {
        if (!shouldBePrinted())
            return;

        printDuration();
        flush();
    }

private:
    const NodePtr& node;
    const int lvl;
    const bool colorUp;
    std::stringstream stream;

    bool shouldBePrinted() const;
    void printInfo();
    void printDuration();
    void flush() const;
};

// use heap allocation instead of stack to align with PERF macro (to have proper destruction order)
#define VERBOSE(...) const auto verbose = std::unique_ptr<Verbose>(new Verbose(__VA_ARGS__));
}   // namespace intel_cpu
}   // namespace ov
#else
#define VERBOSE(...)
#endif // CPU_DEBUG_CAPS
