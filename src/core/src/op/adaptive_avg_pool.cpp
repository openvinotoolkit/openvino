// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/adaptive_avg_pool.hpp"

#include "adaptive_avg_pool_shape_inference.hpp"
#include "itt.hpp"

namespace ov {

op::v8::AdaptiveAvgPool::AdaptiveAvgPool(const Output<Node>& data, const Output<Node>& output_shape)
    : Op({data, output_shape}) {
    constructor_validate_and_infer_types();
}

void op::v8::AdaptiveAvgPool::validate_and_infer_types() {
    OV_OP_SCOPE(v8_AdaptiveAvgPool_validate_and_infer_types);

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

std::shared_ptr<Node> op::v8::AdaptiveAvgPool::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v8_AdaptiveAvgPool_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<v8::AdaptiveAvgPool>(new_args.at(0), new_args.at(1));
}

}  // namespace ov
