// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/tan.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/tan.hpp"

using namespace std;
using namespace ngraph;

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

OPENVINO_SUPPRESS_DEPRECATED_START
namespace tanop {
namespace {
struct Evaluate : ov::element::NoAction<bool> {
    using ov::element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count) {
        ngraph::runtime::reference::tan(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
        return true;
    }
};

bool evaluate_tan(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count) {
    out->set_unary(arg0);

    using namespace ov::element;
    return IfTypeOf<i32, i64, u32, u64, f16, f32>::apply<Evaluate>(arg0->get_element_type(), arg0, out, count);
}
}  // namespace
}  // namespace tanop

bool op::Tan::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_Tan_evaluate);
    return tanop::evaluate_tan(inputs[0], outputs[0], shape_size(inputs[0]->get_shape()));
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
