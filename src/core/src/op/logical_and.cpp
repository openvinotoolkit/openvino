// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/logical_and.hpp"

#include "itt.hpp"
#include "openvino/reference/and.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {
LogicalAnd::LogicalAnd(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseLogical(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> LogicalAnd::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_LogicalAnd_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<LogicalAnd>(new_args.at(0), new_args.at(1), get_autob());
}

bool LogicalAnd::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_LogicalAnd_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    outputs[0].set_shape(infer_broadcast_shape(this, inputs));

    if (inputs[0].get_element_type() == element::boolean) {
        using T = fundamental_type_for<element::boolean>;
        reference::logical_and(inputs[0].data<const T>(),
                               inputs[1].data<const T>(),
                               outputs[0].data<T>(),
                               inputs[0].get_shape(),
                               inputs[1].get_shape(),
                               get_autob());
        return true;
    } else {
        return false;
    }
}

bool LogicalAnd::has_evaluate() const {
    OV_OP_SCOPE(v1_LogicalAnd_has_evaluate);
    return get_input_element_type(0) == element::boolean;
}
}  // namespace v1
}  // namespace op
}  // namespace ov
