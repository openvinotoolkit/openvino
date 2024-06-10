// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qkv_proj.hpp"

#include "transformations/itt.hpp"
namespace ov {
namespace intel_cpu {

void QKVProjectionNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(QKVProjection_validate_and_infer_types);
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this, input_size == 4);

    const auto& ishape = get_input_partial_shape(0);
    const auto& itype = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this, ishape.rank().is_static() && ishape.rank() == 3, "feature shape rank must be 3");

    set_output_size(3);

    auto oshape0 = ishape;
    auto oshape1 = ishape;
    auto oshape2 = ishape;
    oshape0[oshape0.size()-1] = get_input_partial_shape(1)[0];
    oshape1[oshape1.size()-1] = get_input_partial_shape(2)[0];
    oshape2[oshape2.size()-1] = get_input_partial_shape(3)[0];
    set_output_type(0, itype, oshape0);
    set_output_type(1, itype, oshape1);
    set_output_type(2, itype, oshape2);
}

std::shared_ptr<Node> QKVProjectionNode::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(QKVProjection_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<QKVProjectionNode>(new_args);
}
}  // namespace intel_cpu
}  // namespace ov
