// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/add.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace common_translators {

using namespace ov::op;
using namespace std;

OutputVector translate_atan2_util(const NodeContext& context, const Output<Node>& lhs, const Output<Node>& rhs) {
    const double pi_val = atan(1.0) * 4;

    auto div = context.mark_node(make_shared<v1::Divide>(lhs, rhs));
    auto atan = context.mark_node(make_shared<v0::Atan>(div));

    // create some constants to adjust result according to quadrant.
    auto zero = context.mark_node(v0::Constant::create(ov::element::i32, Shape{}, {0}));
    auto pi = context.mark_node(v0::Constant::create(ov::element::f64, Shape{}, {pi_val}));
    auto half_pi = context.mark_node(v0::Constant::create(ov::element::f64, Shape{}, {pi_val / 2}));
    auto neg_half_pi = context.mark_node(v0::Constant::create(ov::element::f64, Shape{}, {-pi_val / 2}));

    zero = context.mark_node(make_shared<v1::ConvertLike>(zero, rhs));
    pi = context.mark_node(make_shared<v1::ConvertLike>(pi, rhs));
    half_pi = context.mark_node(make_shared<v1::ConvertLike>(half_pi, rhs));
    neg_half_pi = context.mark_node(make_shared<v1::ConvertLike>(neg_half_pi, rhs));

    //  x > 0, no adjustment needed
    auto x_greater_than_zero = context.mark_node(make_shared<v1::Greater>(rhs, zero));

    // x < 0 and y >= 0, need to plus pi
    auto y_greater_equal_zero = context.mark_node(make_shared<v1::GreaterEqual>(lhs, zero));
    auto x_less_than_zero = context.mark_node(make_shared<v1::Less>(rhs, zero));
    auto add_pi_condition = context.mark_node(make_shared<v1::LogicalAnd>(x_less_than_zero, y_greater_equal_zero));

    // x < 0 and y < 0, need to minus pi
    auto y_less_than_zero = make_shared<v1::Less>(lhs, zero);

    // x = 0 and y > 0, pi/2
    auto x_equal_zero = make_shared<v1::Equal>(rhs, zero);
    auto y_greater_than_zero = make_shared<v1::Greater>(lhs, zero);
    auto half_pi_condition = context.mark_node(make_shared<v1::LogicalAnd>(x_equal_zero, y_greater_than_zero));

    // x = 0 and y < 0, -pi/2
    auto neg_half_pi_condition = context.mark_node(make_shared<v1::LogicalAnd>(x_equal_zero, y_less_than_zero));

    auto special_case_condition =
        context.mark_node(make_shared<v1::LogicalOr>(half_pi_condition, neg_half_pi_condition));

    // x = 0 and y = 0: with IEEE 754 signed-zero handling.
    // IEEE 754 definitions:
    //      atan2(+0,+0)=+0,
    //      atan2(-0,+0)=-0,
    //      atan2(+0,-0)=+π,
    //      atan2(-0,-0)=-π
    // Standard comparison ops cannot distinguish ±0 (-0 == +0 is true, and -0 < 0 is false).
    // Detect sign via reciprocal: 1/(+0)=+inf, 1/(-0)=-inf, then Less(recip, 0) detects negative sign.
    auto y_equal_zero = context.mark_node(make_shared<v1::Equal>(lhs, zero));
    auto both_zero_condition = context.mark_node(make_shared<v1::LogicalAnd>(x_equal_zero, y_equal_zero));

    auto one = context.mark_node(v0::Constant::create(ov::element::f64, Shape{}, {1.0}));
    one = context.mark_node(make_shared<v1::ConvertLike>(one, rhs));

    // for safe division when non-zero, +inf for +0, -inf for -0 when zero
    auto rhs_for_sign = context.mark_node(make_shared<v1::Select>(both_zero_condition, rhs, one));
    auto lhs_for_sign = context.mark_node(make_shared<v1::Select>(both_zero_condition, lhs, one));
    auto recip_x = context.mark_node(make_shared<v1::Divide>(one, rhs_for_sign));
    auto recip_y = context.mark_node(make_shared<v1::Divide>(one, lhs_for_sign));

    auto x_sign_negative = context.mark_node(make_shared<v1::Less>(recip_x, zero));
    auto y_sign_negative = context.mark_node(make_shared<v1::Less>(recip_y, zero));
    auto x_sign_positive = context.mark_node(make_shared<v1::LogicalNot>(x_sign_negative));
    auto y_sign_positive = context.mark_node(make_shared<v1::LogicalNot>(y_sign_negative));

    auto neg_pi = context.mark_node(make_shared<v0::Negative>(pi));
    auto neg_zero = context.mark_node(make_shared<v0::Negative>(zero));

    // atan2(-0, -0) = -π
    auto case_y_neg_x_neg = context.mark_node(make_shared<v1::LogicalAnd>(x_sign_negative, y_sign_negative));
    // atan2(+0, -0) = +π
    auto case_y_pos_x_neg = context.mark_node(make_shared<v1::LogicalAnd>(x_sign_negative, y_sign_positive));
    // atan2(-0, +0) = -0
    auto case_y_neg_x_pos = context.mark_node(make_shared<v1::LogicalAnd>(x_sign_positive, y_sign_negative));
    // atan2(+0, +0) = +0 (default fallback)

    auto signed_zero_result = context.mark_node(make_shared<v1::Select>(case_y_neg_x_neg, neg_pi, zero));
    signed_zero_result = context.mark_node(make_shared<v1::Select>(case_y_pos_x_neg, pi, signed_zero_result));
    signed_zero_result = context.mark_node(make_shared<v1::Select>(case_y_neg_x_pos, neg_zero, signed_zero_result));

    // do adjustment
    auto atan_plus_pi = context.mark_node(make_shared<v1::Add>(atan, pi));
    auto atan_minus_pi = context.mark_node(make_shared<v1::Subtract>(atan, pi));

    // select result
    auto adjusted_case = context.mark_node(make_shared<v1::Select>(add_pi_condition, atan_plus_pi, atan_minus_pi));
    auto special_case = context.mark_node(make_shared<v1::Select>(half_pi_condition, half_pi, neg_half_pi));
    auto adjusted_atan = context.mark_node(make_shared<v1::Select>(x_greater_than_zero, atan, adjusted_case));
    auto result = context.mark_node(make_shared<v1::Select>(special_case_condition, special_case, adjusted_atan));
    result = context.mark_node(make_shared<v1::Select>(both_zero_condition, signed_zero_result, result));

    return {result};
}

OutputVector translate_atan2(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto lhs = context.get_input(0);
    auto rhs = context.get_input(1);

    return translate_atan2_util(context, lhs, rhs);
}

}  // namespace common_translators
}  // namespace frontend
}  // namespace ov
