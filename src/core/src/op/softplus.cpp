// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/softplus.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/softplus.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v4::SoftPlus);

op::v4::SoftPlus::SoftPlus(const Output<Node>& arg) : Op({arg}) {
    constructor_validate_and_infer_types();
}

bool op::v4::SoftPlus::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v4_SoftPlus_visit_attributes);
    return true;
}

void op::v4::SoftPlus::validate_and_infer_types() {
    OV_OP_SCOPE(v4_SoftPlus_validate_and_infer_types);
    const element::Type& input_et = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_et.is_dynamic() || input_et.is_real(),
                          "Input element type must be float. Got: ",
                          input_et);

    set_output_size(1);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<Node> op::v4::SoftPlus::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v4_SoftPlus_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v4::SoftPlus>(new_args.at(0));
}

namespace softplus {
namespace {
template <element::Type_t ET>
inline bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out, const size_t count) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::softplus<T>(arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
    return true;
}

bool evaluate_softplus(const HostTensorPtr& arg, const HostTensorPtr& out) {
    bool rc = true;
    out->set_unary(arg);
    size_t count = shape_size(arg->get_shape());

    switch (arg->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_softplus, bf16, arg, out, count);
        NGRAPH_TYPE_CASE(evaluate_softplus, f16, arg, out, count);
        NGRAPH_TYPE_CASE(evaluate_softplus, f32, arg, out, count);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace softplus

bool op::v4::SoftPlus::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v4_SoftPlus_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, 1) && validate_host_tensor_vector(inputs, 1));
    return softplus::evaluate_softplus(inputs[0], outputs[0]);
}

bool op::v4::SoftPlus::has_evaluate() const {
    OV_OP_SCOPE(v4_SoftPlus_has_evaluate);
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
