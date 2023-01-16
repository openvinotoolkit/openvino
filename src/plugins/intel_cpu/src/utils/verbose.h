// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef CPU_DEBUG_CAPS

#include <node.h>
#include "utils/verbose_node_helper.h"

#include <string>
#include <cstdlib>
#include <sstream>

namespace ov {
namespace intel_cpu {

class Verbose {
public:
    Verbose(const NodePtr& _node, const std::string& _lvl, const int inferCount)
        : node(_node), lvl(atoi(_lvl.c_str())) {
        if (!shouldBePrinted())
            return;
        node->_verboseStorage.cleanup();
        printInfo(inferCount);
    }

    ~Verbose() {
        if (!shouldBePrinted())
            return;

        printLastInfo();
        flush();
    }

private:
    const NodePtr& node;
    const int lvl;
    std::stringstream stream;

    bool shouldBePrinted() const;
    void printInfo(const int inferCount);
    void printLastInfo();
    void flush() const;

    enum Color {
        RED,
        GREEN,
        YELLOW,
        BLUE,
        PURPLE,
        CYAN
    };
    std::string colorize(const Color color, const std::string& str) const;
};

// use heap allocation instead of stack to align with PERF macro (to have proper destruction order)
#define VERBOSE(...) const auto verbose = std::unique_ptr<Verbose>(new Verbose(__VA_ARGS__));
}   // namespace intel_cpu
}   // namespace ov
#else
#define VERBOSE(...)
#endif // CPU_DEBUG_CAPS
