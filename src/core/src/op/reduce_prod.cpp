// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reduce_prod.hpp"

#include <ngraph/validation_util.hpp>

#include "bound_evaluate.hpp"
#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/reduce_logical_or.hpp"
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

    std::cout << "RP " << get_name() << " - " << inputs[0].get_shape() << std::endl;

    const auto reduction_axes = get_normalized_axes_from_tensor(this, inputs[1], inputs[0].get_shape().size());
    outputs[0].set_shape(ov::util::reduce(inputs[0].get_shape(), reduction_axes, get_keep_dims()));

    std::cout << outputs[0].get_shape() << std::endl;

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
#if 0
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
    if (get_input_tensor(0).get_element_type() == element::i64) {
        const auto& bound = get_input_tensor(0).get_upper_value();
        const auto bound_constant =
            std::make_shared<op::v0::Constant>(bound.get_element_type(), bound.get_shape(), bound.data());
        auto max_constant = ov::op::v0::Constant::create(
            element::i64,
            {1},
            {std::numeric_limits<typename element_type_traits<element::i64>::value_type>::max()});
        OutputVector equal(1);

        bool folded = std::make_shared<op::v1::Equal>(bound_constant, max_constant)
                          ->constant_fold(equal, {bound_constant, max_constant});
        OPENVINO_ASSERT(folded);

        auto axes_vector = std::vector<int64_t>(equal[0].get_shape().size());
        std::iota(axes_vector.begin(), axes_vector.end(), 0);
        const auto axes = op::v0::Constant::create(element::i64, {axes_vector.size()}, axes_vector);

        OutputVector all(1);
        folded = std::make_shared<op::v1::ReduceLogicalOr>(equal[0], axes)->constant_fold(all, {equal[0], axes});
        OPENVINO_ASSERT(folded && ov::is_type<op::v0::Constant>(all[0].get_node_shared_ptr()));
        OPENVINO_ASSERT(all[0].get_shape() == Shape{});

        if (std::dynamic_pointer_cast<op::v0::Constant>(all[0].get_node_shared_ptr())->cast_vector<bool>()[0]) {
            static int64_t maxint = max_constant->cast_vector<int64_t>()[0];
            output_values[0] = ov::Tensor(element::i64, {1}, &maxint);
            // max_constant->evaluate(output_values, {});
            return true;
        }
    }

    return default_upper_bound_evaluator(this, output_values);
}

#else
bool ReduceProd::evaluate_upper(ov::TensorVector& output_values) const {
#    if 0
    const auto& bound = get_input_tensor(0).get_upper_value();
    const auto bound_constant =
        std::make_shared<op::v0::Constant>(bound.get_element_type(), bound.get_shape(), bound.data());
    auto max_constant = ngraph::get_constant_max_of_type(get_input_tensor(0).get_element_type());
    OutputVector equal(1);

    bool folded = std::make_shared<op::v1::Equal>(bound_constant, max_constant)
                      ->constant_fold(equal, {bound_constant, max_constant});
    OPENVINO_ASSERT(folded);

    auto axes_vector = std::vector<int64_t>(equal[0].get_shape().size());
    std::iota(axes_vector.begin(), axes_vector.end(), 0);
    const auto axes = op::v0::Constant::create(element::i64, {axes_vector.size()}, axes_vector);

    OutputVector all(1);
    folded = std::make_shared<op::v1::ReduceLogicalOr>(equal[0], axes)->constant_fold(all, {equal[0], axes});
    OPENVINO_ASSERT(folded && ov::is_type<op::v0::Constant>(all[0].get_node_shared_ptr()));
    OPENVINO_ASSERT(all[0].get_shape() == Shape{});

    if (std::dynamic_pointer_cast<op::v0::Constant>(all[0].get_node_shared_ptr())->cast_vector<bool>()[0]) {
        const auto max_constant = op::v0::Constant::create(get_input_tensor(0).get_element_type(), {1}, {0x7fffffffffffffff});
//        bool status = ;
        return reduce_prod::has_positive_bounds_on_data(this) && get_input_tensor(1).has_and_set_bound() &&
               max_constant->evaluate(output_values, {});
        /*
        OPENVINO_SUPPRESS_DEPRECATED_START
        auto input_maximum_value = ngraph::get_constant_max_of_type(input_tensor.get_element_type());
        OPENVINO_SUPPRESS_DEPRECATED_END
        auto input_max = ov::Tensor(input_tensor.get_element_type(), input_tensor.get_shape());
        memcpy(input_max.data(), input_maximum_value->get_data_ptr(), input_max.get_byte_size());
        // dynamic values translation
        auto input_dynamic_mask = ov::Tensor(element::boolean, input_tensor.get_shape());
        auto outputs = ov::TensorVector{input_dynamic_mask};

        bool status = op::v1::Equal().evaluate(outputs, {value, input_max});
        if (!status)
            return status;

        status = op::v1::Select().evaluate(output_values, {input_dynamic_mask, input_max, output_values[0]});
        return status;*/
    }
#    endif
    return reduce_prod::has_positive_bounds_on_data(this) && get_input_tensor(1).has_and_set_bound() &&
           default_upper_bound_evaluator(this, output_values);
}
#endif
}  // namespace v1
}  // namespace op
}  // namespace ov
