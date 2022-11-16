// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/asinh.hpp"

#include <string>
#include <vector>

#include "itt.hpp"
#include "ngraph/op/util/elementwise_args.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/asinh.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(ov::op::v3::Asinh);

op::v3::Asinh::Asinh(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v3::Asinh::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_Asinh_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Asinh>(new_args.at(0));
}

namespace asinhop {
namespace {
template <element::Type_t ET>
inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count) {
    runtime::reference::asinh(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
    return true;
}

bool evaluate_asinh(const HostTensorPtr& arg0, const HostTensorPtr& out) {
    bool rc = true;
    size_t count = shape_size(arg0->get_shape());
    out->set_unary(arg0);

    switch (arg0->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_asinh, i32, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_asinh, i64, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_asinh, u32, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_asinh, u64, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_asinh, f16, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_asinh, f32, arg0, out, count);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace asinhop

bool op::v3::Asinh::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v3_Asinh_evaluate);
    return asinhop::evaluate_asinh(inputs[0], outputs[0]);
}

bool op::v3::Asinh::has_evaluate() const {
    OV_OP_SCOPE(v3_Asinh_has_evaluate);
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
