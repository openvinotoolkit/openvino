// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qsa_indexer_ref_executor.hpp"

#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/qsa_indexer_config.hpp"
#include "nodes/kernels/sparse_attn/qsa_indexer_common.hpp"
#include "nodes/kernels/sparse_attn/qsa_indexer_kernel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "transformations/cpu_opset/x64/op/qsa_indexer.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

bool QSAIndexerRefExecutor::supports(const QSAIndexerConfig& config) {
    const auto precision = config.descs.at(qsa_src_arg(QSAIndexerNode::InputIdx::Q_IDX))->getPrecision();
    return any_of(precision, ov::element::f32, ov::element::f16, ov::element::bf16);
}

void QSAIndexerRefExecutor::execute(const MemoryArgs& memory) {
    node::qsa::run_indexer(node::qsa::ScalarBackend{}, memory, m_attrs.config, m_context);
}

}  // namespace ov::intel_cpu
