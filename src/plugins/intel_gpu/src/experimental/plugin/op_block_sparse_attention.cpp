// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental/block_sparse_attention.hpp"

#include "experimental/primitives/block_sparse_attention.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"

namespace ov::intel_gpu {

static void CreateBlockSparseAttentionOp(ProgramBuilder& p, const std::shared_ptr<ov::op::experimental::BlockSparseAttention>& op) {
    validate_inputs_count(op, {4, 5});

    const auto input_type = op->get_input_element_type(0);
    if (input_type != ov::element::f16) {
        OPENVINO_THROW("BlockSparseAttention Intel GPU CM kernel currently only supports FP16 precision, but got " + input_type.to_string());
    }

    auto inputs = p.GetInputInfo(op);
    const std::string layerName = layer_type_name_ID(op);

    cldnn::block_sparse_attention prim(layerName, inputs, op->get_block_size_q(), op->get_block_size_kv(), op->get_causal());

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(experimental, BlockSparseAttention);

}  // namespace ov::intel_gpu
