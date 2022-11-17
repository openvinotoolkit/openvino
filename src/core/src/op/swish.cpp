// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/swish.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/swish.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v4::Swish);

op::v4::Swish::Swish(const Output<Node>& arg) : Op({arg}) {
    constructor_validate_and_infer_types();
}

op::v4::Swish::Swish(const Output<Node>& arg, const Output<Node>& beta) : Op({arg, beta}) {
    constructor_validate_and_infer_types();
}

bool op::v4::Swish::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v4_Swish_visit_attributes);
    return true;
}

void op::v4::Swish::validate_and_infer_types() {
    OV_OP_SCOPE(v4_Swish_validate_and_infer_types);
    auto inputs_count = input_values().size();
    NODE_VALIDATION_CHECK(this,
                          inputs_count == 1 || inputs_count == 2,
                          "Swish must have 1 or 2 inputs, but it has: ",
                          inputs_count);

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).is_real(),
                          "Swish input tensor must be floating point type(",
                          get_input_element_type(0),
                          ").");

    if (inputs_count == 2) {
        NODE_VALIDATION_CHECK(this,
                              input_value(0).get_element_type() == input_value(1).get_element_type(),
                              "Swish inputs must have the same type but they are: ",
                              input_value(0).get_element_type(),
                              " and ",
                              input_value(1).get_element_type());
        if (get_input_partial_shape(1).rank().is_static()) {
            auto beta_rank = get_input_partial_shape(1).rank().get_length();
            NODE_VALIDATION_CHECK(this,
                                  beta_rank == 0,
                                  "Swish input with beta must be scalar but it has rank: ",
                                  beta_rank);
        }
    }
    set_output_size(1);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<Node> op::v4::Swish::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v4_Swish_clone_with_new_inputs);
    if (new_args.size() == 1) {
        return make_shared<op::v4::Swish>(new_args.at(0));
    } else {
        return make_shared<op::v4::Swish>(new_args.at(0), new_args.at(1));
    }
}

namespace swish {
namespace {
template <element::Type_t ET>
inline bool evaluate(const HostTensorPtr& arg0,
                     const HostTensorPtr& arg1,
                     const HostTensorPtr& out,
                     const size_t count) {
    using T = typename element_type_traits<ET>::value_type;
    if (arg1 != nullptr) {
        runtime::reference::swish<T>(arg0->get_data_ptr<ET>(),
                                     arg1->get_data_ptr<ET>(),
                                     out->get_data_ptr<ET>(),
                                     count);
    } else {
        runtime::reference::swish<T>(arg0->get_data_ptr<ET>(), nullptr, out->get_data_ptr<ET>(), count);
    }
    return true;
}

bool evaluate_swish(const HostTensorVector& inputs, const HostTensorPtr& out) {
    bool rc = true;
    size_t count = shape_size(inputs[0]->get_shape());

    const HostTensorPtr arg0 = inputs[0];
    const HostTensorPtr arg1 = inputs.size() == 2 ? inputs[1] : nullptr;
    out->set_unary(arg0);

    switch (arg0->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_swish, f16, arg0, arg1, out, count);
        NGRAPH_TYPE_CASE(evaluate_swish, f32, arg0, arg1, out, count);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace swish

bool op::v4::Swish::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v4_Swish_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, 1) &&
                 (validate_host_tensor_vector(inputs, 2) || validate_host_tensor_vector(inputs, 1)));
    return swish::evaluate_swish(inputs, outputs[0]);
}

bool op::v4::Swish::has_evaluate() const {
    OV_OP_SCOPE(v4_Swish_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::f16:
    case ngraph::element::f32:
        return true;
    default:
        break;
    }
    return false;
}
