// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/floor_mod.hpp"

#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/reference/floor_mod.hpp"

using namespace std;
using namespace ngraph;

op::v1::FloorMod::FloorMod(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::FloorMod::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_FloorMod_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<FloorMod>(new_args.at(0), new_args.at(1), this->get_autob());
}

OPENVINO_SUPPRESS_DEPRECATED_START
namespace floor_mod {
namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorPtr& arg0,
              const HostTensorPtr& arg1,
              const HostTensorPtr& out,
              const op::AutoBroadcastSpec& broadcast_spec) {
    ov::reference::floor_mod(arg0->get_data_ptr<ET>(),
                             arg1->get_data_ptr<ET>(),
                             out->get_data_ptr<ET>(),
                             arg0->get_shape(),
                             arg1->get_shape(),
                             broadcast_spec);
    return true;
}

bool evaluate_floor_mod(const HostTensorPtr& arg0,
                        const HostTensorPtr& arg1,
                        const HostTensorPtr& out,
                        const op::AutoBroadcastSpec& broadcast_spec) {
    bool rc = true;
    out->set_broadcast(broadcast_spec, arg0, arg1);
    switch (arg0->get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_floor_mod, i8, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_floor_mod, i32, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_floor_mod, i64, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_floor_mod, u8, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_floor_mod, u32, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_floor_mod, u64, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_floor_mod, bf16, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_floor_mod, f16, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_floor_mod, f32, arg0, arg1, out, broadcast_spec);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace floor_mod

bool op::v1::FloorMod::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v1_FloorMod_evaluate);
    return floor_mod::evaluate_floor_mod(inputs[0], inputs[1], outputs[0], get_autob());
}

bool op::v1::FloorMod::has_evaluate() const {
    OV_OP_SCOPE(v1_FloorMod_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::i8:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u8:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::bf16:
    case ngraph::element::f16:
    case ngraph::element::f32:
        return true;
    default:
        break;
    }
    return false;
}

bool op::v1::FloorMod::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_FloorMod_visit_attributes);
    BinaryElementwiseArithmetic::visit_attributes(visitor);
    return true;
}
