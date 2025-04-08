// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/floor.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/floor.hpp"

namespace ov {
namespace op {
namespace floor {

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& arg, Tensor& out, const size_t count) {
        reference::floor(arg.data<const T>(), out.data<T>(), count);
        return true;
    }
};
}  // namespace floor

namespace v0 {

Floor::Floor(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> Floor::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Floor_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Floor>(new_args.at(0));
}

bool Floor::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Floor_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 1);

    const auto& in_shape = inputs[0].get_shape();
    outputs[0].set_shape(in_shape);

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v0_Floor_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, i8, i16, i32, i64, u8, u16, u32, u64),
                                      floor::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      shape_size(in_shape));
}

bool Floor::has_evaluate() const {
    OV_OP_SCOPE(v0_Floor_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::f16:
    case element::f32:
    case element::i8:
    case element::i16:
    case element::i32:
    case element::i64:
    case element::u8:
    case element::u16:
    case element::u32:
    case element::u64:
        return true;
    default:
        return false;
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov
