// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef CPU_DEBUG_CAPS
#    include <node.h>

#    include <charconv>
#    include <cstdlib>
#    include <memory>
#    include <sstream>
#    include <string>

#    include "verbose.h"

namespace ov::intel_cpu {

static int getLevel(const std::string& str) {
    int value = 0;

    std::from_chars(str.data(), str.data() + str.size(), value);

    return value;
}

class Verbose {
public:
    Verbose(const NodePtr& _node, int _lvl)
        : m_node(_node),
          m_lvl(_lvl % 10),
          /* 1,  2,  3,  etc -> no color. 11, 22, 33, etc -> colorize */
          m_colorUp(_lvl != 0) {
        if (!shouldBePrinted()) {
            return;
        }
        printInfo(m_stream, m_node, m_colorUp);
    }

    Verbose(const NodePtr& _node, const std::string& _lvl) : Verbose(_node, getLevel(_lvl)) {}

    ~Verbose() {
        if (!shouldBePrinted()) {
            return;
        }

        printDuration(m_stream, m_node);
        flush();
    }

private:
    const NodePtr& m_node;
    const int m_lvl;
    const bool m_colorUp;
    std::stringstream m_stream;

    bool shouldBePrinted() const;
    void flush() const;
};

// use heap allocation instead of stack to align with PERF macro (to have proper destruction order)
#    define VERBOSE(...) const auto verbose = std::make_unique<Verbose>(__VA_ARGS__);
}  // namespace ov::intel_cpu

#else
#    define VERBOSE(...)
#endif  // CPU_DEBUG_CAPS
