// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/squared_difference.hpp"

#include "itt.hpp"
#include "openvino/reference/squared_difference.hpp"
#include "utils.hpp"

namespace squared_difference {
struct Evaluate : ov::element::NoAction<bool> {
    using ov::element::NoAction<bool>::visit;

    template <ov::element::Type_t ET>
    static result_type visit(const ov::Tensor& in0,
                             const ov::Tensor& in1,
                             ov::Tensor& out,
                             const ov::Shape& shape0,
                             const ov::Shape& shape1,
                             const ov::op::AutoBroadcastSpec& broadcast_spec) {
        using T = typename ov::element_type_traits<ET>::value_type;
        ov::reference::squared_difference(in0.data<const T>(),
                                          in1.data<const T>(),
                                          out.data<T>(),
                                          shape0,
                                          shape1,
                                          broadcast_spec);
        return true;
    }
};
}  // namespace squared_difference

// ------------------------------ v0 -------------------------------------------

ov::op::v0::SquaredDifference::SquaredDifference(const Output<Node>& arg0,
                                                 const Output<Node>& arg1,
                                                 const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::op::v0::SquaredDifference::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_SquaredDifference_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::op::v0::SquaredDifference>(new_args.at(0), new_args.at(1), this->get_autob());
}

bool ov::op::v0::SquaredDifference::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_SquaredDifference_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    outputs[0].set_shape(infer_broadcast_shape(this, inputs));
    using namespace ov::element;
    return IF_TYPE_OF(v0_SquaredDifference_evaluate,
                      OV_PP_ET_LIST(f32),
                      squared_difference::Evaluate,
                      inputs[0].get_element_type(),
                      inputs[0],
                      inputs[1],
                      outputs[0],
                      inputs[0].get_shape(),
                      inputs[1].get_shape(),
                      get_autob());
}

bool ov::op::v0::SquaredDifference::has_evaluate() const {
    OV_OP_SCOPE(v0_SquaredDifference_has_evaluate);
    if (get_input_element_type(0) == element::f32)
        return true;
    return false;
}
