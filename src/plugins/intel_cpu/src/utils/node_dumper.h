// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef CPU_DEBUG_CAPS
#    include <node.h>

#    include "openvino/util/file_util.hpp"
#    include "utils/debug_caps_config.h"

namespace ov {
namespace intel_cpu {

void dumpInputBlobs(const NodePtr& node, const DebugCapsConfig& config, int count = -1);
void dumpOutputBlobs(const NodePtr& node, const DebugCapsConfig& config, int count = -1);

class DumpHelper {
    const NodePtr& node;
    const int count;
    const DebugCapsConfig& config;

public:
    explicit DumpHelper(const NodePtr& _node, const DebugCapsConfig& _config, int _count = -1)
        : node(_node),
          count(_count),
          config(_config) {
        ov::util::create_directory_recursive(config.blobDumpDir);
        dumpInputBlobs(node, config, count);
    }

    ~DumpHelper() {
        dumpOutputBlobs(node, config, count);
    }
};

#    define DUMP(...) DumpHelper __helper##__node(__VA_ARGS__);
}  // namespace intel_cpu
}  // namespace ov
#else  // CPU_DEBUG_CAPS
#    define DUMP(...)
#endif  // CPU_DEBUG_CAPS
