// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_qsa_indexer_ref_executor.hpp"

#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/paged_qsa_indexer_config.hpp"
#include "nodes/kernels/sparse_attn/paged_qsa_indexer_kernel.hpp"
#include "nodes/kernels/sparse_attn/qsa_indexer_common.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

bool PagedQSAIndexerRefExecutor::supports(const PagedQSAIndexerConfig& config) {
    const auto precision =
        config.descs.at(paged_qsa_src_arg(ov::op::internal::PagedQSAIndexer::InputIdx::BLOCK_INDEX_KEYS_CACHE))
            ->getPrecision();
    return any_of(precision, ov::element::f32, ov::element::f16, ov::element::bf16);
}

void PagedQSAIndexerRefExecutor::execute(const MemoryArgs& memory) {
    node::qsa::paged::run_indexer(node::qsa::ScalarBackend{}, memory, m_attrs.config, m_context);
}

}  // namespace ov::intel_cpu
