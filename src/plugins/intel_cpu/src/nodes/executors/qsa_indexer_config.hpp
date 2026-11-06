// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "executor_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "transformations/cpu_opset/x64/op/qsa_indexer.hpp"

namespace ov::intel_cpu {

struct QSAIndexerAttrs {
    ov::intel_cpu::QSAIndexerNode::Config config;
};

using QSAIndexerConfig = executor::Config<QSAIndexerAttrs>;

// The arguments of the executors are the ports of the op: input `i` is ARG_SRC_0 + i and output `i` is ARG_DST_0 + i
// (QSAIndexerNode::InputIdx / OutputIdx).
constexpr int qsa_src_arg(size_t port) {
    return ARG_SRC_0 + static_cast<int>(port);
}

constexpr int qsa_dst_arg(size_t port) {
    return ARG_DST_0 + static_cast<int>(port);
}

}  // namespace ov::intel_cpu
