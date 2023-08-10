// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/validation_util.hpp>

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/util/evaluate_helpers.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/min.hpp"
#include "ngraph/shape_util.hpp"

using namespace std;
using namespace ngraph;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace minop {
namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out, const AxisSet& axes, const bool keep_dims) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    out->set_shape(reduce(arg->get_shape(), axes, keep_dims));
    OPENVINO_SUPPRESS_DEPRECATED_END
    runtime::reference::min(arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), arg->get_shape(), axes);
    return true;
}

bool evaluate_min(const HostTensorPtr& arg, const HostTensorPtr& out, const AxisSet& axes, const bool keep_dims) {
    bool rc = true;
    switch (arg->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_min, i32, arg, out, axes, keep_dims);
        NGRAPH_TYPE_CASE(evaluate_min, i64, arg, out, axes, keep_dims);
        NGRAPH_TYPE_CASE(evaluate_min, u32, arg, out, axes, keep_dims);
        NGRAPH_TYPE_CASE(evaluate_min, u64, arg, out, axes, keep_dims);
        NGRAPH_TYPE_CASE(evaluate_min, f16, arg, out, axes, keep_dims);
        NGRAPH_TYPE_CASE(evaluate_min, f32, arg, out, axes, keep_dims);
        NGRAPH_TYPE_CASE(evaluate_min, i8, arg, out, axes, keep_dims);
        NGRAPH_TYPE_CASE(evaluate_min, u8, arg, out, axes, keep_dims);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace minop

op::v1::ReduceMin::ReduceMin(const Output<Node>& arg, const Output<Node>& reduction_axes, bool keep_dims)
    : ArithmeticReductionKeepDims(arg, reduction_axes, keep_dims) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::ReduceMin::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_ReduceMin_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v1::ReduceMin>(new_args.at(0), new_args.at(1), get_keep_dims());
}

bool op::v1::ReduceMin::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v1_ReduceMin_evaluate);
    OPENVINO_SUPPRESS_DEPRECATED_START
    NGRAPH_CHECK(validate_host_tensor_vector(inputs, 2));
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, 1));

    const auto reduction_axes =
        get_normalized_axes_from_tensor(inputs[1], inputs[0]->get_partial_shape().rank(), get_friendly_name());
    OPENVINO_SUPPRESS_DEPRECATED_END

    return minop::evaluate_min(inputs[0], outputs[0], reduction_axes, get_keep_dims());
}

bool op::v1::ReduceMin::has_evaluate() const {
    OV_OP_SCOPE(v1_ReduceMin_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::i8:
    case ngraph::element::u8:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32:
        return true;
    default:
        break;
    }
    return false;
}

bool op::v1::ReduceMin::evaluate_lower(ov::TensorVector& output_values) const {
    return input_value(1).get_tensor().has_and_set_bound() && default_lower_bound_evaluator(this, output_values);
}

bool op::v1::ReduceMin::evaluate_upper(ov::TensorVector& output_values) const {
    return input_value(1).get_tensor().has_and_set_bound() && default_upper_bound_evaluator(this, output_values);
}
