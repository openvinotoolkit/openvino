// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/reduce_l2.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/util/evaluate_helpers.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/reduce_l2.hpp"
#include "ngraph/shape_util.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v4::ReduceL2);

op::v4::ReduceL2::ReduceL2(const Output<Node>& arg, const Output<Node>& reduction_axes, bool keep_dims)
    : ArithmeticReductionKeepDims(arg, reduction_axes, keep_dims) {
    constructor_validate_and_infer_types();
}

NGRAPH_SUPPRESS_DEPRECATED_START
shared_ptr<Node> op::v4::ReduceL2::get_default_value() const {
    return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
}
NGRAPH_SUPPRESS_DEPRECATED_END

shared_ptr<Node> op::v4::ReduceL2::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v4_ReduceL2_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v4::ReduceL2>(new_args.at(0), new_args.at(1), get_keep_dims());
}

namespace reduce_l2 {
namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out, const AxisSet& axes, bool keep_dims) {
    out->set_shape(reduce(arg->get_shape(), axes, keep_dims));
    runtime::reference::reduce_l2(arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), arg->get_shape(), axes);
    return true;
}

bool evaluate_reduce_l2(const HostTensorPtr& arg, const HostTensorPtr& out, const AxisSet& axes, bool keep_dims) {
    bool rc = true;
    switch (arg->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_reduce_l2, bf16, arg, out, axes, keep_dims);
        NGRAPH_TYPE_CASE(evaluate_reduce_l2, f16, arg, out, axes, keep_dims);
        NGRAPH_TYPE_CASE(evaluate_reduce_l2, f32, arg, out, axes, keep_dims);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace reduce_l2

bool op::v4::ReduceL2::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v4_ReduceL2_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(inputs, 2));
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, 1));

    const auto reduction_axes =
        get_normalized_axes_from_tensor(inputs[1], inputs[0]->get_partial_shape().rank(), get_friendly_name());

    return reduce_l2::evaluate_reduce_l2(inputs[0], outputs[0], reduction_axes, get_keep_dims());
}

bool op::v4::ReduceL2::has_evaluate() const {
    OV_OP_SCOPE(v4_ReduceL2_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::bf16:
    case ngraph::element::f16:
    case ngraph::element::f32:
        return true;
    default:
        break;
    }
    return false;
}
