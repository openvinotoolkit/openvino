// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/rdft.hpp"

#include <memory>

#include "itt.hpp"
#include "rdft_shape_inference.hpp"

ov::op::v9::RDFT::RDFT(const Output<Node>& data, const Output<Node>& axes) : FFTBase(data, axes) {
    constructor_validate_and_infer_types();
}

ov::op::v9::RDFT::RDFT(const Output<Node>& data, const Output<Node>& axes, const Output<Node>& signal_size)
    : FFTBase(data, axes, signal_size) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::op::v9::RDFT::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v9_RDFT_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() == 2 || new_args.size() == 3, "Number of inputs must be 2 or 3");

    if (new_args.size() == 2) {
        return std::make_shared<ov::op::v9::RDFT>(new_args.at(0), new_args.at(1));
    }

    return std::make_shared<ov::op::v9::RDFT>(new_args.at(0), new_args.at(1), new_args.at(2));
}

void ov::op::v9::RDFT::validate_and_infer_types() {
    OV_OP_SCOPE(v9_RDFT_validate_and_infer_types);

    validate_types();

    std::vector<ov::PartialShape> input_shapes;

    const auto& data = get_input_partial_shape(0);
    const auto& axes = get_input_partial_shape(1);
    if (input_values().size() == 2) {
        input_shapes = {data, axes};
    } else {
        const auto& signal_size = get_input_partial_shape(2);
        input_shapes = {data, axes, signal_size};
    }

    const auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}
