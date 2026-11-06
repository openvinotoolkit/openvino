// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "executor_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"

namespace ov::intel_cpu {

struct PagedQSAIndexerAttrs {
    ov::op::internal::PagedQSAIndexer::Config config;
};

using PagedQSAIndexerConfig = executor::Config<PagedQSAIndexerAttrs>;

/// @brief Fixed output order of ov::op::internal::PagedQSAIndexer (not part of the op's public API).
struct PagedQSAIndexerOutputIdx {
    static constexpr size_t SEL_INDICES = 0;  ///< [tokens, h_sel, block_topk]
    static constexpr size_t SEL_COUNT = 1;    ///< [tokens, h_sel]

    static constexpr size_t COUNT = 2;
};

// The arguments of the executors are the ports of the op: input `i` is ARG_SRC_0 + i and output `i` is ARG_DST_0 + i
// (PagedQSAIndexer::InputIdx / PagedQSAIndexerOutputIdx).
constexpr int paged_qsa_src_arg(size_t port) {
    return ARG_SRC_0 + static_cast<int>(port);
}

constexpr int paged_qsa_dst_arg(size_t port) {
    return ARG_DST_0 + static_cast<int>(port);
}

}  // namespace ov::intel_cpu
