// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reduce_logical_and.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/logical_reduction.hpp"

namespace ov {
namespace op {
namespace reduce_and {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& in, Tensor& out, const AxisSet& reduction_axes) {
        using T = fundamental_type_for<ET>;
        reference::reduce_logical_and(in.data<const T>(), out.data<T>(), in.get_shape(), reduction_axes);
        return true;
    }
};
}  // namespace reduce_and

namespace v1 {
ReduceLogicalAnd::ReduceLogicalAnd(const Output<Node>& data, const Output<Node>& reduction_axes, const bool keep_dims)
    : LogicalReductionKeepDims(data, reduction_axes, keep_dims) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> ReduceLogicalAnd::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_ReduceLogicalAnd_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ReduceLogicalAnd>(new_args.at(0), new_args.at(1), get_keep_dims());
}

bool ReduceLogicalAnd::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_ReduceLogicalAnd_evaluate);
    OPENVINO_ASSERT(inputs.size() == 2);
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto reduction_axes = ov::util::try_get_normalized_axis_set(inputs[1], inputs[0].get_shape().size(), *this);
    outputs[0].set_shape(ov::util::reduce(inputs[0].get_shape(), reduction_axes, get_keep_dims()));

    using namespace ov::element;
    return IF_TYPE_OF(v1_ReduceLogicalAnd_evaluate,
                      boolean,
                      reduce_and::Evaluate,
                      inputs[0].get_element_type(),
                      inputs[0],
                      outputs[0],
                      reduction_axes);
}

bool ReduceLogicalAnd::has_evaluate() const {
    OV_OP_SCOPE(v1_ReduceLogicalAnd_has_evaluate);
    return get_input_element_type(0) == element::boolean && get_input_element_type(1).is_integral_number();
}
}  // namespace v1
}  // namespace op
}  // namespace ov
