// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/hsigmoid.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/reference/hsigmoid.hpp"

using namespace std;
using namespace ngraph;

op::v5::HSigmoid::HSigmoid(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

bool op::v5::HSigmoid::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v5_HSigmoid_visit_attributes);
    return true;
}

shared_ptr<Node> op::v5::HSigmoid::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v5_HSigmoid_clone_with_new_inputs);
    return make_shared<op::v5::HSigmoid>(new_args.at(0));
}

OPENVINO_SUPPRESS_DEPRECATED_START
namespace {
template <element::Type_t ET>
inline bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out, const size_t count) {
    using T = typename element_type_traits<ET>::value_type;

    ov::reference::hsigmoid<T>(arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
    return true;
}

bool evaluate_hsigmoid(const HostTensorPtr& arg, const HostTensorPtr& out) {
    bool rc = true;
    size_t count = shape_size(arg->get_shape());
    out->set_unary(arg);

    switch (arg->get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_hsigmoid, bf16, arg, out, count);
        OPENVINO_TYPE_CASE(evaluate_hsigmoid, f16, arg, out, count);
        OPENVINO_TYPE_CASE(evaluate_hsigmoid, f32, arg, out, count);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace

bool op::v5::HSigmoid::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v5_HSigmoid_evaluate);
    OPENVINO_SUPPRESS_DEPRECATED_START
    OPENVINO_ASSERT(validate_host_tensor_vector(outputs, 1) && validate_host_tensor_vector(inputs, 1));
    OPENVINO_SUPPRESS_DEPRECATED_END
    return evaluate_hsigmoid(inputs[0], outputs[0]);
}

bool op::v5::HSigmoid::has_evaluate() const {
    OV_OP_SCOPE(v5_HSigmoid_has_evaluate);
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
