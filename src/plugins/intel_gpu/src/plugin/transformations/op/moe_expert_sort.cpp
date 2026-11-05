// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/moe_expert_sort.hpp"

namespace ov::intel_gpu::op {

MoEExpertSort::MoEExpertSort(const ov::Output<ov::Node>& topk_indices, size_t num_experts) : ov::op::Op({topk_indices}), m_num_experts(num_experts) {
    constructor_validate_and_infer_types();
}

void MoEExpertSort::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 1, "MoEExpertSort expects 1 input, got ", get_input_size());
    NODE_VALIDATION_CHECK(this, get_input_element_type(0).is_integral_number(), "MoEExpertSort expects integer expert ids");
    NODE_VALIDATION_CHECK(this, m_num_experts > 0, "MoEExpertSort expects num_experts > 0");

    const auto& in_shape = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this, in_shape.rank().compatible(2), "MoEExpertSort expects rank 2 input, got ", in_shape);

    ov::Dimension rows = ov::Dimension::dynamic();
    if (in_shape.rank().is_static()) {
        rows = in_shape[0] * in_shape[1];
    }
    set_output_type(0, ov::element::i32, ov::PartialShape{rows});
    set_output_type(1, ov::element::i32, ov::PartialShape{static_cast<int64_t>(m_num_experts)});
    set_output_type(2, ov::element::i32, ov::PartialShape{rows});
}

std::shared_ptr<ov::Node> MoEExpertSort::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<MoEExpertSort>(new_args.at(0), m_num_experts);
}

bool MoEExpertSort::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("num_experts", m_num_experts);
    return true;
}

}  // namespace ov::intel_gpu::op
