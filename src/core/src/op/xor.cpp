// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/xor.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/reference/xor.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace logxor {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& arg0,
                             const Tensor& arg1,
                             Tensor& out,
                             const Shape& shape0,
                             const Shape& shape1,
                             const AutoBroadcastSpec& broadcast_spec) {
        using T = typename element_type_traits<ET>::value_type;
        reference::logical_xor(arg0.data<const T>(),
                               arg1.data<const T>(),
                               out.data<T>(),
                               shape0,
                               shape1,
                               broadcast_spec);
        return true;
    }
};

namespace {
bool input_supported_type(const element::Type& et) {
    return et == element::boolean;
}

bool evaluate(const Node* const op, TensorVector& outputs, const TensorVector& inputs) {
    OPENVINO_ASSERT(outputs.size() == 1);

    outputs[0].set_shape(infer_broadcast_shape(op, inputs));
    using namespace ov::element;
    return IF_TYPE_OF(Xor_evaluate,
                      boolean,
                      logxor::Evaluate,
                      inputs[0].get_element_type(),
                      inputs[0],
                      inputs[1],
                      outputs[0],
                      inputs[0].get_shape(),
                      inputs[1].get_shape(),
                      op->get_autob());
}
}  // namespace
}  // namespace logxor

namespace v0 {
Xor::Xor(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseLogical(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> Xor::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Xor_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Xor>(new_args.at(0), new_args.at(1), this->get_autob());
}

bool Xor::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Xor_evaluate);

    return logxor::evaluate(this, outputs, inputs);
}

bool Xor::has_evaluate() const {
    OV_OP_SCOPE(v0_Xor_has_evaluate);
    return logxor::input_supported_type(get_input_element_type(0));
}
}  // namespace v0

namespace v1 {
LogicalXor::LogicalXor(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseLogical(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> LogicalXor::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_LogicalXor_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<LogicalXor>(new_args.at(0), new_args.at(1), this->get_autob());
}

bool LogicalXor::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_LogicalXor_evaluate);

    return logxor::evaluate(this, outputs, inputs);
}

bool LogicalXor::has_evaluate() const {
    OV_OP_SCOPE(v1_LogicalXor_has_evaluate);
    return logxor::input_supported_type(get_input_element_type(0));
}
}  // namespace v1
}  // namespace op
}  // namespace ov
