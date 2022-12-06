// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/transpose.hpp"

#include "itt.hpp"
#include "ngraph/runtime/reference/transpose.hpp"
#include "ngraph/validation_util.hpp"
#include "transpose_shape_inference.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v1::Transpose);

op::v1::Transpose::Transpose(const Output<Node>& arg, const Output<Node>& input_order) : Op({arg, input_order}) {
    constructor_validate_and_infer_types();
}

bool op::v1::Transpose::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_Transpose_visit_attributes);
    return true;
}

void op::v1::Transpose::validate_and_infer_types() {
    OV_OP_SCOPE(v1_Transpose_validate_and_infer_types);
    const auto& input_order_et = get_input_element_type(ORDER);
    NODE_VALIDATION_CHECK(this,
                          input_order_et.is_dynamic() || input_order_et.is_integral_number(),
                          "Input order must have an integral number element type.");

    const auto& input_order_shape = get_input_partial_shape(ORDER);
    NODE_VALIDATION_CHECK(this, input_order_shape.rank().compatible(1), "Input order must be a vector.");

    const auto& arg_shape = get_input_partial_shape(ARG);
    NODE_VALIDATION_CHECK(
        this,
        input_order_shape.compatible(ov::PartialShape{arg_shape.rank()}) ||
            (input_order_shape.is_static() && input_order_shape.rank() == 1 && input_order_shape[0] == 0),
        "Input order must have shape [n], where n is the rank of arg.");

    set_input_is_relevant_to_shape(ORDER);

    std::vector<ov::PartialShape> input_shapes{arg_shape, input_order_shape};
    std::vector<ov::PartialShape> output_shapes(OUT_COUNT, ov::PartialShape{});

    shape_infer(this, input_shapes, output_shapes);

    set_output_size(output_shapes.size());
    set_output_type(ARG, get_input_element_type(ARG), output_shapes[ARG_T]);
}

shared_ptr<Node> op::v1::Transpose::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Transpose_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::Transpose>(new_args[ARG], new_args[ORDER]);
}

bool op::v1::Transpose::evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const {
    OV_OP_SCOPE(v1_Transpose_evaluate);

    const auto& order = input_values[ORDER];
    OPENVINO_ASSERT(order->get_element_type().is_integral_number(),
                    "Transpose axis element type has to be integral data type.");

    const auto& arg = input_values[ARG];
    std::vector<int64_t> axes_order = host_tensor_2_vector<int64_t>(order);
    auto out_shape = calc_output_shape(this, arg->get_shape(), axes_order);

    auto& out = output_values[ARG_T];
    out->set_shape(out_shape);
    out->set_element_type(arg->get_element_type());
    ngraph::runtime::reference::transpose(arg->get_data_ptr<char>(),
                                          out->get_data_ptr<char>(),
                                          arg->get_shape(),
                                          arg->get_element_type().size(),
                                          axes_order.data(),
                                          out_shape);
    return true;
}

bool op::v1::Transpose::has_evaluate() const {
    OV_OP_SCOPE(v1_Transpose_has_evaluate);
    return get_input_element_type(1).is_integral_number();
}

bool op::v1::Transpose::evaluate_lower(const HostTensorVector& output_values) const {
    return get_input_tensor(ORDER).has_and_set_bound() && default_lower_bound_evaluator(this, output_values);
}

bool op::v1::Transpose::evaluate_upper(const HostTensorVector& output_values) const {
    return get_input_tensor(ORDER).has_and_set_bound() && default_upper_bound_evaluator(this, output_values);
}

bool op::v1::Transpose::evaluate_label(TensorLabelVector& output_labels) const {
    return get_input_tensor(ORDER).has_and_set_bound() && default_label_evaluator(this, output_labels);
}
