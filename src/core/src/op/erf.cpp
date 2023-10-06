// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/erf.hpp"

#include "itt.hpp"
#include "ngraph/log.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/util.hpp"
#include "openvino/reference/erf.hpp"

using namespace std;
using namespace ngraph;

bool ngraph::op::v0::Erf::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Erf_visit_attributes);
    return true;
}

shared_ptr<Node> op::Erf::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Erf_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Erf>(new_args.at(0));
}

op::Erf::Erf(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

OPENVINO_SUPPRESS_DEPRECATED_START
namespace erfop {
namespace {
template <element::Type_t ET>
inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count) {
    using T = typename element_type_traits<ET>::value_type;
    ov::reference::erf<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
    return true;
}

bool evaluate_erf(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count) {
    bool rc = true;
    out->set_unary(arg0);

    switch (arg0->get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_erf, i32, arg0, out, count);
        OPENVINO_TYPE_CASE(evaluate_erf, i64, arg0, out, count);
        OPENVINO_TYPE_CASE(evaluate_erf, u32, arg0, out, count);
        OPENVINO_TYPE_CASE(evaluate_erf, u64, arg0, out, count);
        OPENVINO_TYPE_CASE(evaluate_erf, f16, arg0, out, count);
        OPENVINO_TYPE_CASE(evaluate_erf, f32, arg0, out, count);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace erfop

bool op::Erf::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_Erf_evaluate);
    return erfop::evaluate_erf(inputs[0], outputs[0], shape_size(inputs[0]->get_shape()));
}

bool op::Erf::has_evaluate() const {
    OV_OP_SCOPE(v0_Erf_has_evaluate);
    switch (get_input_element_type(0)) {
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
