// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/variadic_split.hpp"

#include <numeric>

#include "bound_evaluate.hpp"
#include "compare.hpp"
#include "itt.hpp"
#include "ngraph/validation_util.hpp"
#include "openvino/reference/slice.hpp"
#include "variadic_split_shape_inference.hpp"

using namespace std;
using namespace ngraph;

op::v1::VariadicSplit::VariadicSplit(const Output<Node>& data,
                                     const Output<Node>& axis,
                                     const Output<Node>& split_lengths)
    : Op({data, axis, split_lengths}) {
    constructor_validate_and_infer_types();
}

bool ngraph::op::v1::VariadicSplit::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_VariadicSplit_visit_attributes);
    return true;
}

void ngraph::op::v1::VariadicSplit::validate_and_infer_types() {
    OV_OP_SCOPE(v1_VariadicSplit_validate_and_infer_types);
    for (size_t i = 0; i < get_input_size(); ++i) {
        set_input_is_relevant_to_value(i);
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto input_shapes = get_node_input_partial_shapes(*this);
    OPENVINO_SUPPRESS_DEPRECATED_END
    const auto output_shapes = shape_infer(this, input_shapes);

    const auto& data_type = get_input_element_type(0);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        set_output_type(i, data_type, output_shapes[i]);
    }
}

shared_ptr<Node> op::v1::VariadicSplit::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_VariadicSplit_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::VariadicSplit>(new_args.at(0), new_args.at(1), new_args.at(2));
}

OPENVINO_SUPPRESS_DEPRECATED_START
namespace variadic_split {
namespace {
inline bool evaluate(const HostTensorPtr& in,
                     const HostTensorPtr& out,
                     const Coordinate& lower_bounds,
                     const Coordinate& upper_bounds) {
    const auto& output_shape = out->get_shape();
    const auto has_nonzero_dims = std::none_of(output_shape.begin(), output_shape.end(), ov::cmp::Equal<size_t>(0));

    if (has_nonzero_dims) {
        ov::reference::slice(in->get_data_ptr<const char>(),
                             out->get_data_ptr<char>(),
                             in->get_shape(),
                             lower_bounds,
                             upper_bounds,
                             Strides(lower_bounds.size(), 1),
                             out->get_shape(),
                             in->get_element_type().size());
        return true;
    }
    return false;
}
}  // namespace
}  // namespace variadic_split

bool op::v1::VariadicSplit::evaluate_variadic_split(const HostTensorVector& inputs,
                                                    const HostTensorVector& outputs) const {
    const auto& data_tensor = inputs[0];
    const auto& axis_tensor = inputs[1];
    const auto& split_lengths_tensor = inputs[2];
    OPENVINO_ASSERT(axis_tensor->get_element_type().is_integral_number(),
                    "axis element type is not integral data type");
    OPENVINO_ASSERT(split_lengths_tensor->get_element_type().is_integral_number(),
                    "split_lengths element type is not integral data type");

    OPENVINO_SUPPRESS_DEPRECATED_START
    int64_t axis = host_tensor_2_vector<int64_t>(axis_tensor)[0];
    axis = ngraph::normalize_axis(this, axis, data_tensor->get_partial_shape().rank());
    OPENVINO_SUPPRESS_DEPRECATED_END

    std::vector<ov::PartialShape> input_shapes = {data_tensor->get_partial_shape(),
                                                  axis_tensor->get_partial_shape(),
                                                  split_lengths_tensor->get_partial_shape()};
    auto output_shapes = shape_infer(this, input_shapes, make_tensor_accessor(inputs));

    const auto data_shape = data_tensor->get_shape();
    std::vector<size_t> lower_bounds(data_shape.size(), 0);
    std::vector<size_t> upper_bounds = data_shape;
    upper_bounds[axis] = 0;

    size_t split_pos = 0;
    for (const auto& output : outputs) {
        ov::Shape output_shape = output_shapes[split_pos++].get_shape();
        upper_bounds[axis] += output_shape[axis];
        output->set_shape(output_shape);
        variadic_split::evaluate(data_tensor, output, lower_bounds, upper_bounds);
        lower_bounds.at(axis) = upper_bounds.at(axis);
    }

    return true;
}
bool op::v1::VariadicSplit::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v1_VariadicSplit_evaluate);
    return evaluate_variadic_split(inputs, outputs);
}

bool op::v1::VariadicSplit::has_evaluate() const {
    OV_OP_SCOPE(v1_VariadicSplit_has_evaluate);
    return get_input_element_type(1).is_integral_number() && get_input_element_type(2).is_integral_number();
}

bool op::v1::VariadicSplit::has_axis_and_splits_bound_set() const {
    for (size_t i = 1; i < get_input_size(); ++i) {
        if (!get_input_tensor(i).has_and_set_bound()) {
            return false;
        }
    }
    return true;
}

bool op::v1::VariadicSplit::evaluate_lower(ov::TensorVector& output_values) const {
    OV_OP_SCOPE(v1_Split_evaluate_lower);

    return has_axis_and_splits_bound_set() && default_lower_bound_evaluator(this, output_values);
}

bool op::v1::VariadicSplit::evaluate_upper(ov::TensorVector& output_values) const {
    OV_OP_SCOPE(v1_Split_evaluate_upper);

    return has_axis_and_splits_bound_set() && default_upper_bound_evaluator(this, output_values);
}

bool op::v1::VariadicSplit::evaluate_label(TensorLabelVector& output_labels) const {
    OPENVINO_SUPPRESS_DEPRECATED_START
    return has_axis_and_splits_bound_set() && default_label_evaluator(this, output_labels);
    OPENVINO_SUPPRESS_DEPRECATED_END
}
