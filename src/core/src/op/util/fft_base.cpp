// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/fft_base.hpp"

#include "fft_base_shape_inference.hpp"
#include "itt.hpp"

ov::op::util::FFTBase::FFTBase(const Output<Node>& data, const Output<Node>& axes) : Op({data, axes}) {}

ov::op::util::FFTBase::FFTBase(const Output<Node>& data, const Output<Node>& axes, const Output<Node>& signal_size)
    : Op({data, axes, signal_size}) {}

bool ov::op::util::FFTBase::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(util_FFTBase_visit_attributes);
    return true;
}

void ov::op::util::FFTBase::validate_types() {
    OV_OP_SCOPE(util_FFTBase_validate_types);

    size_t num_of_inputs = get_input_size();
    NODE_VALIDATION_CHECK(this, num_of_inputs == 2 || num_of_inputs == 3, "FFT op must have 2 or 3 inputs.");

    element::Type input_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          input_et == element::f32 || input_et == element::f16 || input_et == element::bf16 ||
                              input_et == element::dynamic,
                          "FFT op input element type must be f32, f16, or bf16");

    element::Type axes_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          axes_et == element::i64 || axes_et == element::i32 || axes_et == element::dynamic,
                          "FFT op axes element type must be i32 or i64");

    if (num_of_inputs == 3) {
        element::Type signal_size_et = get_input_element_type(2);
        NODE_VALIDATION_CHECK(
            this,
            signal_size_et == element::i64 || signal_size_et == element::i32 || signal_size_et == element::dynamic,
            "FFT op signal_size element type must be i32 or i64");
    }
}

void ov::op::util::FFTBase::validate_and_infer_types() {
    OV_OP_SCOPE(util_FFTBase_validate_and_infer_types);

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
