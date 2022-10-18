// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/ceiling.hpp"

#include "itt.hpp"
#include "ngraph/op/util/eval_copy.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/ceiling.hpp"
#include "ngraph/runtime/reference/copy.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v0::Ceiling);

op::Ceiling::Ceiling(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::Ceiling::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Ceiling_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Ceiling>(new_args.at(0));
}

namespace ceiling {
namespace {
// function used by TYPE_CASE
template <element::Type_t ET>
inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::ceiling<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
    return true;
}

// function used by COPY_TENSOR
template <element::Type_t ET>
inline bool copy_tensor(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count) {
    runtime::reference::copy(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
    return true;
}

bool evaluate_ceiling(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count) {
    bool rc = true;
    out->set_unary(arg0);

    switch (arg0->get_element_type()) {
        NGRAPH_COPY_TENSOR(evaluate_ceiling, i8, arg0, out, count);
        NGRAPH_COPY_TENSOR(evaluate_ceiling, i16, arg0, out, count);
        NGRAPH_COPY_TENSOR(evaluate_ceiling, i32, arg0, out, count);
        NGRAPH_COPY_TENSOR(evaluate_ceiling, i64, arg0, out, count);
        NGRAPH_COPY_TENSOR(evaluate_ceiling, u8, arg0, out, count);
        NGRAPH_COPY_TENSOR(evaluate_ceiling, u16, arg0, out, count);
        NGRAPH_COPY_TENSOR(evaluate_ceiling, u32, arg0, out, count);
        NGRAPH_COPY_TENSOR(evaluate_ceiling, u64, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_ceiling, f16, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_ceiling, f32, arg0, out, count);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace ceiling

bool op::Ceiling::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_Ceiling_evaluate);
    return ceiling::evaluate_ceiling(inputs[0], outputs[0], shape_size(get_output_shape(0)));
}

bool op::Ceiling::has_evaluate() const {
    OV_OP_SCOPE(v0_Ceiling_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::i8:
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u8:
    case ngraph::element::u16:
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
