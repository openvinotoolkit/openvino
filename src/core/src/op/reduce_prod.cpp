// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reduce_prod.hpp"

#include "bound_evaluate.hpp"
#include "element_visitor.hpp"
#include "itt.hpp"
#include "ngraph/validation_util.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/util/axes_util.hpp"
#include "openvino/reference/reduce_prod.hpp"

namespace ov {
namespace op {
namespace reduce_prod {
namespace {
bool has_positive_bounds_on_data(const Node* const op) {
    const auto& lb = op->get_input_tensor(0).get_lower_value();
    const auto& ub = op->get_input_tensor(0).get_upper_value();

    return lb && ub && tensor_is_non_negative(lb) && tensor_is_non_negative(ub);
}
}  // namespace

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& in0, Tensor& out, const AxisSet& reduction_axes) {
        using T = fundamental_type_for<ET>;
        reference::reduce_prod(in0.data<const T>(), out.data<T>(), in0.get_shape(), reduction_axes);
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
    return reduce_prod::has_positive_bounds_on_data(this) && get_input_tensor(1).has_and_set_bound() &&
           default_lower_bound_evaluator(this, output_values);
}

bool ReduceProd::evaluate_upper(ov::TensorVector& output_values) const {
    if (!reduce_prod::has_positive_bounds_on_data(this) || !get_input_tensor(1).has_and_set_bound())
        return false;
    // We need to cover a corner case: if an Upper Bound comes from ShapeOf and contains
    // dynamic dimension (-1) - it has a value 0x7FFFFFFFFFFFFFFF, which points on
    // a maximum possible value. For example, Upper Bound of shape [-1, 12] is
    // [0x7FFFFFFFFFFFFFFF, 12].
    // In such case we shouldn't evaluate a real ReduceProd because it'll cause an
    // overflow and returns wrong value. We should return an Upper Bound as for [-1],
    // which will be evaluated as [0x7FFFFFFFFFFFFFFF]
    // In case dimensions has a zero dimension - it should return 0 in any case
    if (tensor_has_max_value(get_input_tensor(0).get_upper_value()) &&
        !tensor_has_zero_value(get_input_tensor(0).get_upper_value())) {
        OPENVINO_SUPPRESS_DEPRECATED_START
        auto max_constant = ngraph::get_constant_max_of_type(get_output_element_type(0));
        OPENVINO_SUPPRESS_DEPRECATED_END
        OPENVINO_ASSERT(max_constant->get_byte_size() <= output_values[0].get_byte_size());
        memcpy(output_values[0].data(), max_constant->get_data_ptr(), max_constant->get_byte_size());
        return true;
    }

    return default_upper_bound_evaluator(this, output_values);
}

}  // namespace v1
}  // namespace op
}  // namespace ov
