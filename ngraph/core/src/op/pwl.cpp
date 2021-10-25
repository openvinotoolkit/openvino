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
    // TODO
}

namespace pwlop {
template <element::Type_t ET>
inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count,
    const HostTensorPtr& arg1, const HostTensorPtr& arg2, const size_t segments_number,
    const HostTensorPtr& arg3, const size_t knots_number) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::pwl<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count,
        arg1->get_data_ptr<double>(), arg2->get_data_ptr<double>(), segments_number,
        arg3->get_data_ptr<double>(), knots_number);
    return true;
}

bool evaluate_pwl(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count,
    const HostTensorPtr& arg1, const HostTensorPtr& arg2, const size_t segments_number,
    const HostTensorPtr& arg3, const size_t knots_number) {
    bool rc = true;
    out->set_unary(arg0);

    switch (arg0->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_pwl, i32, arg0, out, count, arg1, arg2, segments_number, arg3, knots_number);
        NGRAPH_TYPE_CASE(evaluate_pwl, i64, arg0, out, count, arg1, arg2, segments_number, arg3, knots_number);
        NGRAPH_TYPE_CASE(evaluate_pwl, u32, arg0, out, count, arg1, arg2, segments_number, arg3, knots_number);
        NGRAPH_TYPE_CASE(evaluate_pwl, u64, arg0, out, count, arg1, arg2, segments_number, arg3, knots_number);
        NGRAPH_TYPE_CASE(evaluate_pwl, f16, arg0, out, count, arg1, arg2, segments_number, arg3, knots_number);
        NGRAPH_TYPE_CASE(evaluate_pwl, f32, arg0, out, count, arg1, arg2, segments_number, arg3, knots_number);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace pwlop

bool op::v0::Pwl::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    NGRAPH_OP_SCOPE(v0_Pwl_evaluate);
    return pwlop::evaluate_pwl(inputs[0], outputs[0], shape_size(get_input_shape(0)),
        inputs[1], inputs[2], shape_size(get_input_shape(1)),
        inputs[3], shape_size(get_input_shape(3)));
}

bool op::v0::Pwl::has_evaluate() const {
    NGRAPH_OP_SCOPE(v0_Pwl_has_evaluate);
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
