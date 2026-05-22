// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/atan2.hpp"
#include "openvino/op/util/broadcast_base.hpp"

namespace ov::intel_gpu::op {

Atan2::Atan2(const ov::Output<Node>& y, const ov::Output<Node>& x) : Op({y, x}) {
    validate_and_infer_types();
}

void Atan2::validate_and_infer_types() {
    const auto& y_shape = get_input_partial_shape(0);
    const auto& x_shape = get_input_partial_shape(1);
    OPENVINO_ASSERT(get_input_element_type(0) == get_input_element_type(1),
                    "Atan2 requires both inputs to have the same element type");

    ov::PartialShape out_shape = y_shape;
    OPENVINO_ASSERT(ov::PartialShape::broadcast_merge_into(out_shape, x_shape, ov::op::AutoBroadcastType::NUMPY),
                    "Atan2 inputs are not broadcast-compatible");

    set_output_type(0, get_input_element_type(0), out_shape);
}

std::shared_ptr<ov::Node> Atan2::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<Atan2>(new_args.at(0), new_args.at(1));
}

}  // namespace ov::intel_gpu::op
