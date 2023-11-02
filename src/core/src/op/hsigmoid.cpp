// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/hsigmoid.hpp"

#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"  // tbr
#include "ngraph/validation_util.hpp"      // tbr
#include "openvino/reference/hsigmoid.hpp"

namespace ov {
namespace op {
namespace v5 {
HSigmoid::HSigmoid(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> HSigmoid::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v5_HSigmoid_clone_with_new_inputs);
    return std::make_shared<HSigmoid>(new_args.at(0));
}

OPENVINO_SUPPRESS_DEPRECATED_START
namespace {
template <element::Type_t ET>
inline bool evaluate(const ngraph::HostTensorPtr& arg, const ngraph::HostTensorPtr& out, const size_t count) {
    using T = typename element_type_traits<ET>::value_type;

    ov::reference::hsigmoid<T>(arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
    return true;
}

bool evaluate_hsigmoid(const ngraph::HostTensorPtr& arg, const ngraph::HostTensorPtr& out) {
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

bool HSigmoid::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v5_HSigmoid_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 1);
    return evaluate_hsigmoid(inputs[0], outputs[0]);
}

bool HSigmoid::has_evaluate() const {
    OV_OP_SCOPE(v5_HSigmoid_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::bf16:
    case element::f16:
    case element::f32:
        return true;
    default:
        return false;
    }
}
}  // namespace v5
}  // namespace op
}  // namespace ov
