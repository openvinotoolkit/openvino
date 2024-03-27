// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "causal_mask_preprocess.hpp"

#include <algorithm>

#include "transformations/itt.hpp"

ov::intel_cpu::CausalMaskPreprocessNode::CausalMaskPreprocessNode(const OutputVector& args, const Config& cfg) : Op(args), m_config(cfg) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::intel_cpu::CausalMaskPreprocessNode::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(CausalMaskPreprocessNode_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::intel_cpu::CausalMaskPreprocessNode>(new_args, m_config);
}

void ov::intel_cpu::CausalMaskPreprocessNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(CausalMaskPreprocessNode_validate_and_infer_types);
    if (m_config.type == "CausalMaskPreprocess") {
        // inputs:
        //   0: attention_mask            : i64[N, kv_len]
        //                             0 means mask-out, 1 means attends to
        //   1: batch_size (size_Gather)  : i32[1]
        //   2: cache_positions  i32[q_len];
        //   3: kvLen            i32[1];
        // outputs
        //   0: causal mask for SDPA : f32[batch_size, 1, q_len, kvLen]

        auto batch_size = Dimension::dynamic();
        auto q_len = get_input_partial_shape(2)[0];
        auto kv_len = Dimension::dynamic();
        set_output_type(0, ov::element::f32, {batch_size, 1, q_len, kv_len});
        return;
    }
    NODE_VALIDATION_CHECK(this, false, "unsupported type : ", m_config.type);
}

bool ov::intel_cpu::CausalMaskPreprocessNode::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(CausalMaskPreprocessNode_visit_attributes);
    visitor.start_structure("config");
    visitor.on_attribute("type", m_config.type);
    visitor.finish_structure();
    return true;
}
