// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/slice.hpp"

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/slice.hpp"
#include "slice_shape_inference.hpp"

namespace ov {
namespace op {
namespace {
std::vector<int64_t> default_axes(const size_t n) {
    std::vector<int64_t> axes;
    axes.reserve(n);
    std::generate_n(std::back_inserter(axes), n, SeqGen<int64_t>(0));
    return axes;
}

bool slice_bound_check(const ov::Node* const node) {
    return ov::have_node_inputs_bounds_set(node, 1, node->get_input_size() - 1);
}

bool slice_no_axes(const Node* const node) {
    return node->get_input_size() < 5;
}
}  // namespace

namespace v8 {
using ov::op::v0::Constant;

Slice::Slice(const Output<Node>& data, const Output<Node>& start, const Output<Node>& stop, const Output<Node>& step)
    : Op({data, start, stop, step}) {
    constructor_validate_and_infer_types();
}

Slice::Slice(const Output<Node>& data,
             const Output<Node>& start,
             const Output<Node>& stop,
             const Output<Node>& step,
             const Output<Node>& axes)
    : Op({data, start, stop, step, axes}) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Constant> Slice::get_default_const_axes(const Output<Node>& start) const {
    const auto& start_pshape = start.get_partial_shape();
    // Static case
    if (start_pshape.is_static() && start_pshape.size() == 1) {
        const auto axes = default_axes(static_cast<size_t>(start_pshape[0].get_length()));
        return Constant::create(element::i64, start_pshape.get_shape(), axes);
    } else {
        // Dynamic case
        return {};
    }
}  // namespace ov

void Slice::validate_and_infer_types() {
    OV_OP_SCOPE(v8_Slice_validate_and_infer_types);

    if (slice_no_axes(this)) {
        if (auto axes_const = get_default_const_axes(input_value(1))) {
            set_argument(4, axes_const);
        }
    }

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    set_input_is_relevant_to_shape(0);
    for (size_t i = 1; i < get_input_size(); ++i) {
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(i).is_integral_number(),
                              "Slice `",
                              slice::shape_names[i - 1],
                              "` input type must be integer.");
        set_input_is_relevant_to_shape(i);
    }

    set_output_type(0, get_input_element_type(0), output_shapes.front());
}

std::shared_ptr<Node> Slice::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v8_Slice_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 4) {
        return std::make_shared<Slice>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
    } else {
        return std::make_shared<Slice>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), new_args.at(4));
    }
}

bool Slice::has_evaluate() const {
    OV_OP_SCOPE(v8_Slice_has_evaluate);

    const auto valid_integral_type = [](const element::Type& et) -> bool {
        switch (et) {
        case element::i8:
        case element::i16:
        case element::i32:
        case element::i64:
        case element::u8:
        case element::u16:
        case element::u32:
        case element::u64:
            return true;
        default:
            return false;
        }
    };

    return valid_integral_type(get_input_element_type(1)) &&
           (slice_no_axes(this) || valid_integral_type(get_input_element_type(4)));
}

bool Slice::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v8_Slice_evaluate);

    const auto output_shapes =
        shape_infer(this, ov::util::get_tensors_partial_shapes(inputs), make_tensor_accessor(inputs));
    outputs[0].set_shape(output_shapes.front().to_shape());

    const auto starts = ov::get_tensor_data_as<int64_t>(inputs[1]);
    const auto steps = ov::get_tensor_data_as<int64_t>(inputs[3]);
    const auto& axes = slice_no_axes(this) ? default_axes(starts.size()) : ov::get_tensor_data_as<int64_t>(inputs[4]);

    reference::slice(static_cast<const char*>(inputs[0].data()),
                     inputs[0].get_shape(),
                     static_cast<char*>(outputs[0].data()),
                     outputs[0].get_shape(),
                     inputs[0].get_element_type().size(),
                     starts,
                     steps,
                     axes);
    return true;
}

bool Slice::evaluate_lower(ov::TensorVector& output_values) const {
    return slice_bound_check(this) && default_lower_bound_evaluator(this, output_values);
}

bool Slice::evaluate_upper(ov::TensorVector& output_values) const {
    return slice_bound_check(this) && default_upper_bound_evaluator(this, output_values);
}

bool Slice::evaluate_symbol(TensorSymbolVector& output_symbols) const {
    return slice_bound_check(this) && ov::util::default_symbol_evaluator(this, output_symbols);
}
}  // namespace v8
}  // namespace op
}  // namespace ov
