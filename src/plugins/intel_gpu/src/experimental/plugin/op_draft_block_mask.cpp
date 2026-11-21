// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental/draft_block_mask.hpp"

#include "experimental/primitives/draft_block_mask.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"

namespace ov::intel_gpu {

static void CreateDraftBlockMaskOp(ProgramBuilder& p, const std::shared_ptr<ov::op::experimental::DraftBlockMask>& op) {
    validate_inputs_count(op, {3, 5});

    auto inputs = p.GetInputInfo(op);
    const std::string layerName = layer_type_name_ID(op);

    if (op->get_input_size() == 5) {
        cldnn::draft_block_mask prim(layerName, inputs, op->get_num_heads(), op->get_batch_size());
        p.add_primitive(*op, prim);
    } else {
        cldnn::draft_block_mask prim(layerName, inputs, op->get_num_heads(), op->get_seqlen(), op->get_topk(), op->get_batch_size());
        p.add_primitive(*op, prim);
    }
}

REGISTER_FACTORY_IMPL(experimental, DraftBlockMask);

}  // namespace ov::intel_gpu
