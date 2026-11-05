// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/moe_scatter_reduce.hpp"

namespace ov::intel_gpu::op {

MoEScatterReduce::MoEScatterReduce(const ov::Output<ov::Node>& data, const ov::Output<ov::Node>& weights, const ov::Output<ov::Node>& row_lut)
    : ov::op::Op({data, weights, row_lut}) {
    constructor_validate_and_infer_types();
}

void MoEScatterReduce::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 3, "MoEScatterReduce expects 3 inputs, got ", get_input_size());
    NODE_VALIDATION_CHECK(this, get_input_element_type(2) == ov::element::i32, "MoEScatterReduce expects i32 row_lut");

    const auto& data_shape = get_input_partial_shape(0);
    const auto& weights_shape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(this, data_shape.rank().compatible(2), "MoEScatterReduce expects rank 2 data, got ", data_shape);
    NODE_VALIDATION_CHECK(this, weights_shape.rank().compatible(2), "MoEScatterReduce expects rank 2 weights, got ", weights_shape);
    NODE_VALIDATION_CHECK(this, get_input_partial_shape(2).rank().compatible(1), "MoEScatterReduce expects rank 1 row_lut");

    const auto tokens = weights_shape.rank().is_static() ? weights_shape[0] : ov::Dimension::dynamic();
    const auto hidden = data_shape.rank().is_static() ? data_shape[1] : ov::Dimension::dynamic();
    set_output_type(0, get_input_element_type(0), ov::PartialShape{tokens, hidden});
}

std::shared_ptr<ov::Node> MoEScatterReduce::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<MoEScatterReduce>(new_args.at(0), new_args.at(1), new_args.at(2));
}

}  // namespace ov::intel_gpu::op
