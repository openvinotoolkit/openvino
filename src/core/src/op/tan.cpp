// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/tan.hpp"

#include "itt.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/tan.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v0::Tan);

op::Tan::Tan(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::Tan::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Tan_visit_attributes);
    return true;
}

shared_ptr<Node> op::Tan::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Tan_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Tan>(new_args.at(0));
}

namespace tanop {
namespace {
template <element::Type_t ET>
inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::tan<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
    return true;
}

bool evaluate_tan(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count) {
    bool rc = true;
    out->set_unary(arg0);

    switch (arg0->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_tan, i32, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_tan, i64, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_tan, u32, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_tan, u64, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_tan, f16, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_tan, f32, arg0, out, count);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace tanop

bool op::Tan::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_Tan_evaluate);
    return tanop::evaluate_tan(inputs[0], outputs[0], shape_size(get_output_shape(0)));
}

bool op::Tan::has_evaluate() const {
    OV_OP_SCOPE(v0_Tan_has_evaluate);
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
