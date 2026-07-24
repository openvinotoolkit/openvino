// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/atan2_decomposition.hpp"

#include <cmath>
#include <memory>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/atan2.hpp"
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
#include "openvino/pass/node_registry.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

namespace ov::pass {

Atan2Decomposition::Atan2Decomposition() {
    MATCHER_SCOPE(Atan2Decomposition);
    auto atan2_pattern = pattern::wrap_type<ov::op::v17::Atan2>();
    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        auto m_atan2 = m.get_match_root();

        if (transformation_callback(m_atan2)) {
            return false;
        }

        const auto lhs = m_atan2->input_value(0);  // y
        const auto rhs = m_atan2->input_value(1);  // x

        const double pi_val = std::atan(1.0) * 4;

        NodeRegistry rg;
        auto div = rg.make<v1::Divide>(lhs, rhs);
        auto atan = rg.make<v0::Atan>(div);

        // Constants for quadrant adjustment, created in f64 then converted to match input type.
        auto zero_const = rg.add(v0::Constant::create(ov::element::i32, ov::Shape{}, {0}));
        auto pi_const = rg.add(v0::Constant::create(ov::element::f64, ov::Shape{}, {pi_val}));
        auto half_pi_const = rg.add(v0::Constant::create(ov::element::f64, ov::Shape{}, {pi_val / 2}));
        auto neg_half_pi_const = rg.add(v0::Constant::create(ov::element::f64, ov::Shape{}, {-pi_val / 2}));

        auto zero = rg.make<v1::ConvertLike>(zero_const, rhs);
        auto pi = rg.make<v1::ConvertLike>(pi_const, rhs);
        auto half_pi = rg.make<v1::ConvertLike>(half_pi_const, rhs);
        auto neg_half_pi = rg.make<v1::ConvertLike>(neg_half_pi_const, rhs);

        // x > 0: no adjustment needed.
        auto x_greater_than_zero = rg.make<v1::Greater>(rhs, zero);

        // x < 0 and y >= 0: add pi.
        auto y_greater_equal_zero = rg.make<v1::GreaterEqual>(lhs, zero);
        auto x_less_than_zero = rg.make<v1::Less>(rhs, zero);
        auto add_pi_condition = rg.make<v1::LogicalAnd>(x_less_than_zero, y_greater_equal_zero);

        // x < 0 and y < 0: subtract pi.
        auto y_less_than_zero = rg.make<v1::Less>(lhs, zero);

        // x = 0 and y > 0: pi/2.
        auto x_equal_zero = rg.make<v1::Equal>(rhs, zero);
        auto y_greater_than_zero = rg.make<v1::Greater>(lhs, zero);
        auto half_pi_condition = rg.make<v1::LogicalAnd>(x_equal_zero, y_greater_than_zero);

        // x = 0 and y < 0: -pi/2.
        auto neg_half_pi_condition = rg.make<v1::LogicalAnd>(x_equal_zero, y_less_than_zero);

        auto special_case_condition = rg.make<v1::LogicalOr>(half_pi_condition, neg_half_pi_condition);

        // x = 0 and y = 0: IEEE 754 signed-zero handling.
        // Standard comparison cannot distinguish +0 from -0.
        // Use reciprocal: 1/(+0) = +inf, 1/(-0) = -inf, then Less(recip, 0) detects negative sign.
        auto y_equal_zero = rg.make<v1::Equal>(lhs, zero);
        auto both_zero_condition = rg.make<v1::LogicalAnd>(x_equal_zero, y_equal_zero);

        auto one_const = rg.add(v0::Constant::create(ov::element::f64, ov::Shape{}, {1.0}));
        auto one = rg.make<v1::ConvertLike>(one_const, rhs);

        auto rhs_for_sign = rg.make<v1::Select>(both_zero_condition, rhs, one);
        auto lhs_for_sign = rg.make<v1::Select>(both_zero_condition, lhs, one);
        auto recip_x = rg.make<v1::Divide>(one, rhs_for_sign);
        auto recip_y = rg.make<v1::Divide>(one, lhs_for_sign);

        auto x_sign_negative = rg.make<v1::Less>(recip_x, zero);
        auto y_sign_negative = rg.make<v1::Less>(recip_y, zero);
        auto x_sign_positive = rg.make<v1::LogicalNot>(x_sign_negative);
        auto y_sign_positive = rg.make<v1::LogicalNot>(y_sign_negative);

        auto neg_pi = rg.make<v0::Negative>(pi);
        auto neg_zero = rg.make<v0::Negative>(zero);

        // atan2(-0, -0) = -pi
        auto case_y_neg_x_neg = rg.make<v1::LogicalAnd>(x_sign_negative, y_sign_negative);
        // atan2(+0, -0) = +pi
        auto case_y_pos_x_neg = rg.make<v1::LogicalAnd>(x_sign_negative, y_sign_positive);
        // atan2(-0, +0) = -0
        auto case_y_neg_x_pos = rg.make<v1::LogicalAnd>(x_sign_positive, y_sign_negative);
        // atan2(+0, +0) = +0 (default)

        auto sz_step1 = rg.make<v1::Select>(case_y_neg_x_neg, neg_pi, zero);
        auto sz_step2 = rg.make<v1::Select>(case_y_pos_x_neg, pi, sz_step1);
        auto signed_zero_result = rg.make<v1::Select>(case_y_neg_x_pos, neg_zero, sz_step2);

        // Quadrant adjustment.
        auto atan_plus_pi = rg.make<v1::Add>(atan, pi);
        auto atan_minus_pi = rg.make<v1::Subtract>(atan, pi);

        // Select final result.
        auto adjusted_case = rg.make<v1::Select>(add_pi_condition, atan_plus_pi, atan_minus_pi);
        auto special_case = rg.make<v1::Select>(half_pi_condition, half_pi, neg_half_pi);
        auto adjusted_atan = rg.make<v1::Select>(x_greater_than_zero, atan, adjusted_case);
        auto result_step1 = rg.make<v1::Select>(special_case_condition, special_case, adjusted_atan);
        auto result = rg.make<v1::Select>(both_zero_condition, signed_zero_result, result_step1);

        result->set_friendly_name(m_atan2->get_friendly_name());
        copy_runtime_info(m_atan2, rg.get());
        replace_node(m_atan2, result);

        return true;
    };
    auto m = std::make_shared<pattern::Matcher>(atan2_pattern, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::pass
