// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/variadic_split.hpp"

#include <numeric>

#include "bound_evaluate.hpp"
#include "compare.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/slice.hpp"
#include "variadic_split_shape_inference.hpp"

namespace ov {
namespace op {
namespace variadic_split {
namespace {

bool has_axis_and_splits_bound_set(const Node* const node) {
    return have_node_inputs_bounds_set(node, 1, 2);
}

bool evaluate(TensorVector& outputs, const TensorVector& inputs) {
    const auto& data_tensor = inputs[0];
    const auto& axis_tensor = inputs[1];
    const auto axis =
        ov::util::normalize(get_tensor_data_as<int64_t>(axis_tensor).front(), data_tensor.get_shape().size());

    ov::Coordinate upper_bounds(data_tensor.get_shape());
    ov::Coordinate lower_bounds(upper_bounds.size());
    upper_bounds[axis] = 0;

    const Strides default_strides(upper_bounds.size(), 1);
    constexpr auto is_zero_dim = ov::cmp::Equal<size_t>(0);

    for (auto& output : outputs) {
        const auto& out_shape = output.get_shape();
        upper_bounds[axis] += out_shape[axis];

        if (std::none_of(out_shape.cbegin(), out_shape.cend(), is_zero_dim)) {
            reference::slice(static_cast<const char*>(data_tensor.data()),
                             static_cast<char*>(output.data()),
                             data_tensor.get_shape(),
                             lower_bounds,
                             upper_bounds,
                             default_strides,
                             out_shape,
                             data_tensor.get_element_type().size());
        }

        lower_bounds[axis] = upper_bounds[axis];
    }

    return true;
}
}  // namespace
}  // namespace variadic_split

namespace v1 {
VariadicSplit::VariadicSplit(const Output<Node>& data, const Output<Node>& axis, const Output<Node>& split_lengths)
    : Op({data, axis, split_lengths}) {
    constructor_validate_and_infer_types();
}

void VariadicSplit::validate_and_infer_types() {
    OV_OP_SCOPE(v1_VariadicSplit_validate_and_infer_types);
    for (size_t i = 0; i < get_input_size(); ++i) {
        set_input_is_relevant_to_value(i);
    }

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    const auto& data_type = get_input_element_type(0);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        set_output_type(i, data_type, output_shapes[i]);
    }
}

std::shared_ptr<Node> VariadicSplit::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_VariadicSplit_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<VariadicSplit>(new_args.at(0), new_args.at(1), new_args.at(2));
}

bool VariadicSplit::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_VariadicSplit_evaluate);

    if (inputs[1].get_element_type().is_integral_number() && inputs[2].get_element_type().is_integral_number()) {
        const auto output_shapes =
            shape_infer(this, ov::util::get_tensors_partial_shapes(inputs), make_tensor_accessor(inputs));
        OPENVINO_ASSERT(outputs.size() == output_shapes.size());

        auto out_partial_shape = output_shapes.cbegin();
        for (auto& output : outputs) {
            output.set_shape(out_partial_shape->to_shape());
            ++out_partial_shape;
        }

        return variadic_split::evaluate(outputs, inputs);
    } else {
        return false;
    }
}

bool VariadicSplit::has_evaluate() const {
    OV_OP_SCOPE(v1_VariadicSplit_has_evaluate);
    return get_input_element_type(1).is_integral_number() && get_input_element_type(2).is_integral_number();
}

bool VariadicSplit::evaluate_lower(TensorVector& output_values) const {
    OV_OP_SCOPE(v1_Split_evaluate_lower);
    return variadic_split::has_axis_and_splits_bound_set(this) && default_lower_bound_evaluator(this, output_values);
}

bool VariadicSplit::evaluate_upper(TensorVector& output_values) const {
    OV_OP_SCOPE(v1_Split_evaluate_upper);
    return variadic_split::has_axis_and_splits_bound_set(this) && default_upper_bound_evaluator(this, output_values);
}

bool VariadicSplit::evaluate_symbol(TensorSymbolVector& output_symbols) const {
    return variadic_split::has_axis_and_splits_bound_set(this) &&
           ov::util::default_symbol_evaluator(this, output_symbols);
}
}  // namespace v1
}  // namespace op
}  // namespace ov
