// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/hswish.hpp"

#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "openvino/reference/hswish.hpp"

using namespace ngraph;

namespace ov {
namespace op {
namespace v4 {
HSwish::HSwish(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> HSwish::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v4_HSwish_clone_with_new_inputs);
    return std::make_shared<HSwish>(new_args.at(0));
}

OPENVINO_SUPPRESS_DEPRECATED_START
namespace hswish {
namespace {
template <element::Type_t ET>
inline bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out, const size_t count) {
    using T = typename element_type_traits<ET>::value_type;

    ov::reference::hswish<T>(arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
    return true;
}

bool evaluate_hswish(const HostTensorPtr& arg, const HostTensorPtr& out) {
    bool rc = true;
    size_t count = shape_size(arg->get_shape());
    out->set_unary(arg);

    switch (arg->get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_hswish, bf16, arg, out, count);
        OPENVINO_TYPE_CASE(evaluate_hswish, f16, arg, out, count);
        OPENVINO_TYPE_CASE(evaluate_hswish, f32, arg, out, count);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace hswish

bool HSwish::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v4_HSwish_evaluate);
    OPENVINO_SUPPRESS_DEPRECATED_START
    OPENVINO_ASSERT(validate_host_tensor_vector(outputs, 1) && validate_host_tensor_vector(inputs, 1));
    OPENVINO_SUPPRESS_DEPRECATED_END
    return hswish::evaluate_hswish(inputs[0], outputs[0]);
}

bool HSwish::has_evaluate() const {
    OV_OP_SCOPE(v4_HSwish_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::bf16:
    case ngraph::element::f16:
    case ngraph::element::f32:
        return true;
    default:
        return false;
    }
}
}  // namespace v4
}  // namespace op
}  // namespace ov
