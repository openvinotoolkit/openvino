// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef CPU_DEBUG_CAPS

#include "mkldnn_node.h"

#include <string>
#include <cstdlib>
#include <sstream>

namespace MKLDNNPlugin {

class Verbose {
public:
    Verbose(const MKLDNNNodePtr& _node, const std::string& _lvl)
        : node(_node), lvl(atoi(_lvl.c_str())) {
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
    const MKLDNNNodePtr& node;
    const int lvl;
    std::stringstream stream;

    bool shouldBePrinted() const;
    void printInfo();
    void printDuration();
    void flush() const;
};

// use heap allocation instead of stack to align with PERF macro (to have proper destruction order)
#define VERBOSE(...) const auto verbose = std::unique_ptr<Verbose>(new Verbose(__VA_ARGS__));
} // namespace MKLDNNPlugin
#else
#define VERBOSE(...)
#endif // CPU_DEBUG_CAPS
