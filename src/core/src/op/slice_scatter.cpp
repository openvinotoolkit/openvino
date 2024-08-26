// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/slice_scatter.hpp"

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "slice_scatter_shape_inference.hpp"

namespace ov {
namespace op {
namespace v15 {
SliceScatter::SliceScatter(const Output<Node>& data,
                           const Output<Node>& updates,
                           const Output<Node>& start,
                           const Output<Node>& stop,
                           const Output<Node>& step)
    : Op({data, updates, start, stop, step}) {
    constructor_validate_and_infer_types();
}

SliceScatter::SliceScatter(const Output<Node>& data,
                           const Output<Node>& updates,
                           const Output<Node>& start,
                           const Output<Node>& stop,
                           const Output<Node>& step,
                           const Output<Node>& axes)
    : Op({data, updates, start, stop, step, axes}) {
    constructor_validate_and_infer_types();
}

void SliceScatter::validate_and_infer_types() {
    OV_OP_SCOPE(v15_SliceScatter_validate_and_infer_types);
    for (size_t i = 2; i < get_input_size(); ++i) {
        const auto shapes_element_type = get_input_element_type(i);
        NODE_VALIDATION_CHECK(this,
                              shapes_element_type.is_dynamic() || shapes_element_type.is_integral_number(),
                              "SliceScatter `",
                              slice::shape_names[i - 2],
                              "` input type must be integer.");
    }
    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shape = shape_infer(this, input_shapes);
    set_output_type(0, get_input_element_type(0), output_shape[0]);
}

std::shared_ptr<Node> SliceScatter::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v15_SliceScatter_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 5) {
        return std::make_shared<SliceScatter>(new_args.at(0),
                                              new_args.at(1),
                                              new_args.at(2),
                                              new_args.at(3),
                                              new_args.at(4));
    } else {
        return std::make_shared<SliceScatter>(new_args.at(0),
                                              new_args.at(1),
                                              new_args.at(2),
                                              new_args.at(3),
                                              new_args.at(4),
                                              new_args.at(5));
    }
}
}  // namespace v15
}  // namespace op
}  // namespace ov
