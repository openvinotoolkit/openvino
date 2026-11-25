// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sparse_attention.hpp"

#include <string>

#include "intel_gpu/plugin/program_builder.hpp"
#include "plugin/transformations/convert_sparse_attention.hpp"

// ov::op::v17::SparseAttention is lowered as the GPU SparseSDPA op it is converted to by ConvertSparseAttentionToInternal
// (see CreateSparseSDPAOp). The pass leaves the ops it cannot convert, which are reported here with the reason.

namespace ov::intel_gpu {

static void CreateSparseAttentionOp(ProgramBuilder& /*p*/, const std::shared_ptr<ov::op::v17::SparseAttention>& op) {
    std::string reason;
    if (ConvertSparseAttentionToInternal::make_config(*op, reason)) {
        reason = "it was not converted to SparseSDPA";
    }
    OPENVINO_THROW("[GPU] Cannot lower SparseAttention '", op->get_friendly_name(), "': ", reason);
}

REGISTER_FACTORY_IMPL(v17, SparseAttention);

}  // namespace ov::intel_gpu
