// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef CPU_DEBUG_CAPS
#pragma once

#include "mkldnn_node.h"
#include "config.h"

namespace MKLDNNPlugin {

void dumpInputBlobs(const MKLDNNNodePtr &node, const Config& config, int count = -1);
void dumpOutputBlobs(const MKLDNNNodePtr &node, const Config& config, int count = -1);

class DumpHelper {
    const MKLDNNNodePtr& node;
    const int count;
    const Config& config;

public:
    explicit DumpHelper(const MKLDNNNodePtr& _node, const Config& _config, int _count = -1): node(_node), config(_config), count(_count) {
        dumpInputBlobs(node, config, count);
    }

    ~DumpHelper() {
        dumpOutputBlobs(node, config, count);
    }
};

#define DUMP(...) DumpHelper __helper##__node (__VA_ARGS__);
} // namespace MKLDNNPlugin
#else // CPU_DEBUG_CAPS
#define DUMP(...)
#endif // CPU_DEBUG_CAPS
