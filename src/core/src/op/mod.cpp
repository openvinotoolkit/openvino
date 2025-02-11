// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/mod.hpp"

#include "bound_evaluate.hpp"
#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/core/tensor_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/select.hpp"
#include "openvino/reference/mod.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace mod {
struct Evaluate : ov::element::NoAction<bool> {
    using ov::element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& in0,
                             const Tensor& in1,
                             Tensor& out,
                             const Shape& shape0,
                             const Shape& shape1,
                             const AutoBroadcastSpec& broadcast_spec) {
        using T = typename element_type_traits<ET>::value_type;
        reference::mod(in0.data<const T>(), in1.data<const T>(), out.data<T>(), shape0, shape1, broadcast_spec);
        return true;
    }
};

struct EvaluateBound : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& v_lb,
                             const Tensor& v_ub,
                             const Tensor& m_lb,
                             const Tensor& m_ub,
                             Tensor& out,
                             const bool is_lower) {
        auto v_lb_first = v_lb.data<const T>();
        auto v_lb_last = std::next(v_lb_first, v_lb.get_size());
        auto v_ub_first = v_ub.data<const T>();
        auto m_lb_first = m_lb.data<const T>();
        auto m_ub_first = m_ub.data<const T>();
        auto out_first = out.data<T>();

        if (is_lower) {
            while (v_lb_first != v_lb_last) {
                *out_first++ =
                    reference::func::mod_interval(*v_lb_first++, *v_ub_first++, *m_lb_first++, *m_ub_first++).first;
            }
        } else {
            while (v_lb_first != v_lb_last) {
                *out_first++ =
                    reference::func::mod_interval(*v_lb_first++, *v_ub_first++, *m_lb_first++, *m_ub_first++).second;
            }
        }
        return true;
    }
};

namespace {

/**
 * @brief Get node inputs bounds as TensorVector.
 *
 * The inputs bounds are stored as [lower0, upper0, lower1, upper1].
 *
 * @param op  Pointer to the node.
 * @return Vector with inputs bounds tensors.
 */
TensorVector get_bounds(const Node* const op) {
    auto&& v_bounds = ov::util::evaluate_both_bounds(op->input_value(0));
    auto&& m_bounds = ov::util::evaluate_both_bounds(op->input_value(1));
    return {std::move(v_bounds.first),
            std::move(v_bounds.second),
            std::move(m_bounds.first),
            std::move(m_bounds.second)};
}

/**
 * @brief Check if all bounds in vector are valid (allocated).
 *
 * @param bounds  TensorVector of bounds for check.
 * @return True if bounds area valid otherwise false.
 */
bool are_bounds_valid(const TensorVector& bounds) {
    return std::all_of(bounds.begin(), bounds.end(), [](const Tensor& t) {
        return static_cast<bool>(t);
    });
}

/**
 * @brief Evaluate binary mask of values which cannot be calculated by modulo.
 *
 * @param bounds       Modulo inputs bounds.
 * @return Tensor with binary mask or empty tensor if evaluate failed.
 */
Tensor evaluate_undefined_result_mask(const TensorVector& bounds) {
    const auto eq_op = v1::Equal();
    const auto or_op = v1::LogicalOr();

    const auto& in_et = bounds.front().get_element_type();

    const auto zero_t = ov::util::make_tensor_of_value(in_et, 0);
    const auto max_t = ov::util::make_tensor_of_max_value(in_et);

    const auto& v_ub = bounds[1];
    const auto& m_lb = bounds[2];
    const auto& m_ub = bounds[3];

    auto m_mask = TensorVector{{element::boolean, m_ub.get_shape()}};
    if (!eq_op.evaluate(m_mask, {m_lb, zero_t})) {
        return {};
    }

    auto out_masks = TensorVector{{element::boolean, m_lb.get_shape()}};
    if (!eq_op.evaluate(out_masks, {m_ub, zero_t})) {
        return {};
    }

    auto m_or_inputs = TensorVector{out_masks[0], m_mask[0]};
    or_op.evaluate(m_mask, m_or_inputs);
    if (!eq_op.evaluate(out_masks, {m_lb, max_t})) {
        return {};
    }

    or_op.evaluate(m_mask, m_or_inputs);
    auto v_mask = TensorVector{{element::boolean, v_ub.get_shape()}};
    if (!eq_op.evaluate(v_mask, {v_ub, max_t})) {
        return {};
    }

    out_masks[0].set_shape(ov::op::infer_broadcast_shape(&or_op, v_mask[0].get_shape(), m_mask[0].get_shape()));
    return or_op.evaluate(out_masks, {v_mask[0], m_mask[0]}) ? out_masks[0] : Tensor{};
}

/**
 * @brief Get the inputs bound with valid values only.
 *
 * The values which result modulo to give undefined result are replaced by one.
 * The auto broadcast is applied to have inputs same shape.
 *
 * @param bounds  Modulo operator inputs bounds.
 * @param mask    Mask with undefined result values.
 * @return Vector of bounds tensors.
 */
TensorVector get_bounds_with_valid_values(const TensorVector& bounds, const Tensor& mask) {
    const auto select_op = v1::Select();
    const auto one_t = ov::util::make_tensor_of_value(bounds.front().get_element_type(), 1);

    auto m_bounds = TensorVector();
    m_bounds.reserve(bounds.size());
    std::transform(bounds.cbegin(), bounds.cend(), std::back_inserter(m_bounds), [&](const Tensor& b) -> ov::Tensor {
        auto tmp = TensorVector{{b.get_element_type(), mask.get_shape()}};
        return select_op.evaluate(tmp, {mask, one_t, b}) ? tmp.front() : Tensor{};
    });
    return m_bounds;
}

/**
 * @brief Evaluate modulo upper or lower bound.
 *
 * @param op        Pointer to modulo node.
 * @param outputs   Tensor vector with one tensor to store bounds result.
 * @param is_lower  True to evaluate lower otherwise evaluate upper.
 * @return True if outputs has valid data otherwise false.
 */
bool evaluate_bound(const Node* const op, TensorVector& outputs, bool is_lower) {
    const auto bounds = mod::get_bounds(op);

    if (mod::are_bounds_valid(bounds)) {
        const auto& in_et = bounds[0].get_element_type();

        const auto undefined_result_mask = mod::evaluate_undefined_result_mask(bounds);
        if (!undefined_result_mask) {
            return false;
        }

        // Set inputs values to 1 for undefined results mask (0, inf, etc.)
        const auto m_bounds = mod::get_bounds_with_valid_values(bounds, undefined_result_mask);
        if (!mod::are_bounds_valid(m_bounds)) {
            return false;
        }

        // Evaluate bound.
        outputs[0].set_shape(undefined_result_mask.get_shape());
        using namespace ov::element;
        if (!IfTypeOf<i8, i16, i32, i64, u8, u16, u32, u64>::apply<mod::EvaluateBound>(in_et,
                                                                                       m_bounds[0],
                                                                                       m_bounds[1],
                                                                                       m_bounds[2],
                                                                                       m_bounds[3],
                                                                                       outputs[0],
                                                                                       is_lower)) {
            return false;
        }
        // Set undefined bound value for results which cannot be calculated.
        const auto select_op = v1::Select();
        const auto& undefined_bound =
            is_lower ? ov::util::make_tensor_of_value(in_et, 0) : ov::util::make_tensor_of_max_value(in_et);
        return select_op.evaluate(outputs, {undefined_result_mask, undefined_bound, outputs.front()});
    } else {
        return false;
    }
}
}  // namespace
}  // namespace mod

namespace v1 {
v1::Mod::Mod(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> Mod::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Mod_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Mod>(new_args.at(0), new_args.at(1), this->get_autob());
}

bool Mod::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OV_OP_SCOPE(v1_Mod_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    outputs[0].set_shape(infer_broadcast_shape(this, inputs));
    using namespace ov::element;
    return IF_TYPE_OF(v1_Mod_evaluate,
                      OV_PP_ET_LIST(i8, i16, i32, i64, u8, u16, u32, u64),
                      mod::Evaluate,
                      inputs[0].get_element_type(),
                      inputs[0],
                      inputs[1],
                      outputs[0],
                      inputs[0].get_shape(),
                      inputs[1].get_shape(),
                      get_autob());
}

bool Mod::evaluate_lower(TensorVector& outputs) const {
    OV_OP_SCOPE(v1_Mod_evaluate_lower);
    return mod::evaluate_bound(this, outputs, true);
}

bool Mod::evaluate_upper(TensorVector& outputs) const {
    OV_OP_SCOPE(v1_Mod_evaluate_upper);
    return mod::evaluate_bound(this, outputs, false);
}

bool Mod::has_evaluate() const {
    OV_OP_SCOPE(v1_Mod_has_evaluate);

    switch (get_input_element_type(0)) {
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
}  // namespace v1
}  // namespace op
}  // namespace ov
