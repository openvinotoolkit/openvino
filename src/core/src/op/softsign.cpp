// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/softsign.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/util.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/softsign.hpp"

namespace {
template <ov::element::Type_t ET>
inline bool evaluate(const ov::HostTensorPtr& arg, const ov::HostTensorPtr& out, const size_t count) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::softsign<T>(arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
    return true;
}

bool evaluate_softsign(const ov::HostTensorPtr& arg, const ov::HostTensorPtr& out) {
    bool rc = true;
    size_t count = shape_size(arg->get_shape());

    switch (arg->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_softsign, bf16, arg, out, count);
        NGRAPH_TYPE_CASE(evaluate_softsign, f16, arg, out, count);
        NGRAPH_TYPE_CASE(evaluate_softsign, f32, arg, out, count);
        NGRAPH_TYPE_CASE(evaluate_softsign, f64, arg, out, count);
        default:
            rc = false;
            break;
    }
    return rc;
}

} // namespace

// *** SOFTSIGN OP SET V9 **
BWDCMP_RTTI_DEFINITION(ov::op::v9::SoftSign);

ov::op::v9::SoftSign::SoftSign(const Output<Node> &arg) : Op({arg}) {
    constructor_validate_and_infer_types();
}

bool ov::op::v9::SoftSign::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v9_SoftSign_visit_attributes);
    return true;
}

void ov::op::v9::SoftSign::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v9_SoftSign_validate_and_infer_types);
    const element::Type& input_et = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_et.is_dynamic() || input_et.is_real(),
                          "Input element type must be float. Got: ",
                          input_et);

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<ov::Node> ov::op::v9::SoftSign::clone_with_new_inputs(const OutputVector &new_args) const {
    NGRAPH_OP_SCOPE(v9_SoftSign_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::op::v9::SoftSign>(new_args.at(0));
}

bool ov::op::v9::SoftSign::has_evaluate() const {
    NGRAPH_OP_SCOPE(v9_SoftSign_has_evaluate);
    switch (get_input_element_type(0)) {
        case ngraph::element::bf16:
        case ngraph::element::f16:
        case ngraph::element::f32:
        case ngraph::element::f64:
            return true;
        default:
            break;
    }
    return false;
}

bool ov::op::v9::SoftSign::evaluate(const HostTensorVector &outputs, const HostTensorVector &inputs) const {
    NGRAPH_OP_SCOPE(v9_SoftSign_evaluate);
    NGRAPH_CHECK(ngraph::validate_host_tensor_vector(outputs, 1) && ngraph::validate_host_tensor_vector(inputs, 1));
    outputs[0]->set_unary(inputs[0]);
    return evaluate_softsign(inputs[0], outputs[0]);
}
