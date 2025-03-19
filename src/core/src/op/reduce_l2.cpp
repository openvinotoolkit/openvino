// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reduce_l2.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/reduce_l2.hpp"
#include "reduce_shape_inference.hpp"

namespace ov {
namespace op {

namespace reduce_l2 {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& in0, Tensor& out, const AxisSet& reduction_axes) {
        using T = fundamental_type_for<ET>;
        reference::reduce_l2(in0.data<const T>(), out.data<T>(), in0.get_shape(), reduction_axes);
        return true;
    }
};
}  // namespace reduce_l2
namespace v4 {

ReduceL2::ReduceL2(const Output<Node>& arg, const Output<Node>& reduction_axes, bool keep_dims)
    : ArithmeticReductionKeepDims(arg, reduction_axes, keep_dims) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> ReduceL2::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v4_ReduceL2_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v4::ReduceL2>(new_args.at(0), new_args.at(1), get_keep_dims());
}

bool ReduceL2::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v4_ReduceL2_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 2);

    const auto reduction_axes = ov::util::try_get_normalized_axis_set(inputs[1], inputs[0].get_shape().size(), *this);
    outputs[0].set_shape(ov::util::reduce(inputs[0].get_shape(), reduction_axes, get_keep_dims()));

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v4_ReduceL2_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32),
                                      reduce_l2::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      reduction_axes);
}

bool ReduceL2::has_evaluate() const {
    OV_OP_SCOPE(v4_ReduceL2_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::bf16:
    case element::f16:
    case element::f32:
        return true;
    default:
        return false;
    }
}
}  // namespace v4
}  // namespace op
}  // namespace ov
