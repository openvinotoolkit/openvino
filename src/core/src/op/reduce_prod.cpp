// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reduce_prod.hpp"

#include "bound_evaluate.hpp"
#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/op/util/axes_util.hpp"
#include "openvino/reference/product.hpp"
#include "shape_util.hpp"

namespace ov {
namespace op {
namespace reduce_prod {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& in0, Tensor& out, const AxisSet& reduction_axes) {
        using T = fundamental_type_for<ET>;
        reference::product(in0.data<const T>(), out.data<T>(), in0.get_shape(), reduction_axes);
        return true;
    }
};
}  // namespace reduce_prod
namespace v1 {

ReduceProd::ReduceProd(const Output<Node>& arg, const Output<Node>& reduction_axes, bool keep_dims)
    : ArithmeticReductionKeepDims(arg, reduction_axes, keep_dims) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> ReduceProd::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_ReduceProd_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ReduceProd>(new_args.at(0), new_args.at(1), get_keep_dims());
}

bool ReduceProd::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_ReduceProd_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 2);

    const auto reduction_axes = get_normalized_axes_from_tensor(this, inputs[1], inputs[0].get_shape().size());
    outputs[0].set_shape(ov::util::reduce(inputs[0].get_shape(), reduction_axes, get_keep_dims()));

    using namespace ov::element;
    return IfTypeOf<f16, f32, i32, i64, u32, u64>::apply<reduce_prod::Evaluate>(inputs[0].get_element_type(),
                                                                                inputs[0],
                                                                                outputs[0],
                                                                                reduction_axes);
}

bool ReduceProd::has_evaluate() const {
    OV_OP_SCOPE(v1_ReduceProd_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::i32:
    case element::i64:
    case element::u32:
    case element::u64:
    case element::f16:
    case element::f32:
        return true;
    default:
        return false;
    }
}

bool ReduceProd::evaluate_lower(ov::TensorVector& output_values) const {
    if (!input_value(1).get_tensor().has_and_set_bound())
        return false;

    const auto &lb = input_value(0).get_tensor().get_lower_value(), ub = input_value(0).get_tensor().get_upper_value();
    if (!lb || !ub || !tensor_is_positive(lb) || !tensor_is_positive(ub))
        return false;
    return default_lower_bound_evaluator(this, output_values);
}

bool ReduceProd::evaluate_upper(ov::TensorVector& output_values) const {
    if (!input_value(1).get_tensor().has_and_set_bound())
        return false;

    const auto &lb = input_value(0).get_tensor().get_lower_value(), ub = input_value(0).get_tensor().get_upper_value();
    if (!lb || !ub || !tensor_is_positive(lb) || !tensor_is_positive(ub))
        return false;
    return default_upper_bound_evaluator(this, output_values);
}
}  // namespace v1
}  // namespace op
}  // namespace ov
