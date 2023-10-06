// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/floor.hpp"

#include "itt.hpp"
#include "ngraph/op/util/eval_copy.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/reference/copy.hpp"
#include "openvino/reference/floor.hpp"

using namespace std;
using namespace ngraph;

op::Floor::Floor(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::Floor::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Floor_visit_attributes);
    return true;
}

shared_ptr<Node> op::Floor::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Floor_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Floor>(new_args.at(0));
}

OPENVINO_SUPPRESS_DEPRECATED_START
namespace floorop {
namespace {
// function used by TYPE_CASE
template <element::Type_t ET>
inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count) {
    using T = typename element_type_traits<ET>::value_type;
    ov::reference::floor<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
    return true;
}

// function used by COPY_TENSOR
template <element::Type_t ET>
inline bool copy_tensor(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count) {
    ov::reference::copy(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
    return true;
}

bool evaluate_floor(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count) {
    bool rc = true;
    out->set_unary(arg0);

    switch (arg0->get_element_type()) {
        OPENVINO_COPY_TENSOR(evaluate_floor, i8, arg0, out, count);
        OPENVINO_COPY_TENSOR(evaluate_floor, i16, arg0, out, count);
        OPENVINO_COPY_TENSOR(evaluate_floor, i32, arg0, out, count);
        OPENVINO_COPY_TENSOR(evaluate_floor, i64, arg0, out, count);
        OPENVINO_COPY_TENSOR(evaluate_floor, u8, arg0, out, count);
        OPENVINO_COPY_TENSOR(evaluate_floor, u16, arg0, out, count);
        OPENVINO_COPY_TENSOR(evaluate_floor, u32, arg0, out, count);
        OPENVINO_COPY_TENSOR(evaluate_floor, u64, arg0, out, count);
        OPENVINO_TYPE_CASE(evaluate_floor, f16, arg0, out, count);
        OPENVINO_TYPE_CASE(evaluate_floor, f32, arg0, out, count);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace floorop

bool op::Floor::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_Floor_evaluate);
    return floorop::evaluate_floor(inputs[0], outputs[0], shape_size(inputs[0]->get_shape()));
}

bool op::Floor::has_evaluate() const {
    OV_OP_SCOPE(v0_Floor_has_evaluate);
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
