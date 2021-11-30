// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/pwl.hpp"

#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/pwl.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v0::Pwl);

op::v0::Pwl::Pwl(const Output<Node>& data, const Output<Node>& m, const Output<Node>& b, const Output<Node>& knots)
    : Op({data, m, b, knots}) {
    constructor_validate_and_infer_types();
}

bool op::v0::Pwl::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v0_Pwl_visit_attributes);
    return true;
}

shared_ptr<Node> op::v0::Pwl::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v0_Pwl_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v0::Pwl>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

void op::v0::Pwl::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

template <typename T1, typename T2>
bool op::v0::Pwl::evaluate(const HostTensorVector& outputs,
                           const HostTensorVector& inputs) const {
    outputs[0]->set_unary(inputs[0]);
    OV_SCOPE(ngraph_op, OV_PP_CAT4(region, _, T1, T2)) {
        using A1 = typename element_type_traits<T1::value>::value_type;
        using A2 = typename element_type_traits<T2::value>::value_type;
        ngraph::runtime::reference::pwl(inputs[0]->get_data_ptr<A2>(),
                                        outputs[0]->get_data_ptr<A2>(),
                                        shape_size(get_input_shape(0)),
                                        inputs[1]->get_data_ptr<A1>(),
                                        inputs[2]->get_data_ptr<A1>(),
                                        inputs[3]->get_data_ptr<A1>(),
                                        shape_size(get_input_shape(1)));
        return true;
    }
    return false;
}

bool op::v0::Pwl::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    NGRAPH_OP_SCOPE(v0_Pwl_evaluate);
    return evaluate_pwl(std::tuple<std::integral_constant<element::Type_t, element::f32>,
                                   std::integral_constant<element::Type_t, element::f64>>(),
                        std::tuple<std::integral_constant<element::Type_t, element::i32>,
                                   std::integral_constant<element::Type_t, element::i64>,
                                   std::integral_constant<element::Type_t, element::u32>,
                                   std::integral_constant<element::Type_t, element::u64>,
                                   std::integral_constant<element::Type_t, element::f16>,
                                   std::integral_constant<element::Type_t, element::f32>,
                                   std::integral_constant<element::Type_t, element::f64>>(),
                        outputs,
                        inputs);
}

bool op::v0::Pwl::has_evaluate() const {
    NGRAPH_OP_SCOPE(v0_Pwl_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::i32:
    case element::i64:
    case element::u32:
    case element::u64:
    case element::f16:
    case element::f32:
        return true;
    default:
        break;
    }
    return false;
}
