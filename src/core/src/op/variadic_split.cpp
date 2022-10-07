// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/variadic_split.hpp"

#include <numeric>

#include "itt.hpp"
#include "ngraph/runtime/reference/slice.hpp"
#include "ngraph/validation_util.hpp"
#include "variadic_split_shape_inference.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v1::VariadicSplit);

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
    set_input_is_relevant_to_value(0);
    set_input_is_relevant_to_value(1);
    set_input_is_relevant_to_value(2);

    std::vector<ov::PartialShape> input_shapes = {get_input_partial_shape(0),
                                                  get_input_partial_shape(1),
                                                  get_input_partial_shape(2)};
    std::vector<ov::PartialShape> output_shapes;
    shape_infer(this, input_shapes, output_shapes);

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

namespace variadic_split {
namespace {
inline bool evaluate(const HostTensorPtr& in,
                     const HostTensorPtr& out,
                     const Coordinate& lower_bounds,
                     const Coordinate& upper_bounds) {
    const auto& output_shape = out->get_shape();
    auto has_nonzero_dims = std::all_of(output_shape.begin(), output_shape.end(), [](size_t dim) {
        return dim != 0;
    });

    if (has_nonzero_dims) {
        runtime::reference::slice(in->get_data_ptr<const char>(),
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
    NGRAPH_CHECK(axis_tensor->get_element_type().is_integral_number(), "axis element type is not integral data type");
    NGRAPH_CHECK(split_lengths_tensor->get_element_type().is_integral_number(),
                 "split_lengths element type is not integral data type");

    int64_t axis = host_tensor_2_vector<int64_t>(axis_tensor)[0];
    axis = ngraph::normalize_axis(this, axis, data_tensor->get_partial_shape().rank());

    std::vector<ov::PartialShape> input_shapes = {data_tensor->get_partial_shape(),
                                                  axis_tensor->get_partial_shape(),
                                                  split_lengths_tensor->get_partial_shape()};
    std::vector<ov::PartialShape> output_shapes;
    shape_infer(this, input_shapes, output_shapes, {{1, axis_tensor}, {2, split_lengths_tensor}});

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
