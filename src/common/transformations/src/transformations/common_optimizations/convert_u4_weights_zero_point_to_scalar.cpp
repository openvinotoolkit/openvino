// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_u4_weights_zero_point_to_scalar.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/reference/autobroadcast_binop.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConvertU4WeightsZeroPointToScalar::ConvertU4WeightsZeroPointToScalar() {
    MATCHER_SCOPE(ConvertU4WeightsZeroPointToScalar);
    auto weights_m = pattern::wrap_type<ov::op::v0::Constant>(pattern::type_matches(ov::element::u4));
    auto convert_m = pattern::wrap_type<ov::op::v0::Convert>({weights_m}, pattern::consumers_count(1));

    auto float_zp_predicate = [](ov::Output<ov::Node> output) -> bool {
        return pattern::type_matches_any({ov::element::f32, ov::element::f16})(output) &&
               pattern::consumers_count(1)(output);
    };
    auto float_zero_point_m = pattern::wrap_type<ov::op::v0::Constant>(float_zp_predicate);

    auto u4_zp_predicate = [](ov::Output<ov::Node> output) -> bool {
        return pattern::type_matches(ov::element::u4)(output) && pattern::consumers_count(1)(output);
    };
    auto u4_zero_point_m = pattern::wrap_type<ov::op::v0::Constant>(u4_zp_predicate);
    auto zero_point_convert_m = pattern::wrap_type<ov::op::v0::Convert>({u4_zero_point_m}, float_zp_predicate);

    auto zero_point_m = std::make_shared<pattern::op::Or>(OutputVector{float_zero_point_m, zero_point_convert_m});
    auto subtract_m = pattern::wrap_type<ov::op::v1::Subtract>({convert_m, zero_point_m});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto& pattern_map = m.get_pattern_value_map();
        auto weights = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(weights_m).get_node_shared_ptr());
        std::shared_ptr<ov::op::v0::Constant> zero_point;
        if (pattern_map.count(float_zero_point_m)) {
            const auto& float_zp = pattern_map.at(float_zero_point_m);
            zero_point = ov::as_type_ptr<ov::op::v0::Constant>(float_zp.get_node_shared_ptr());
        } else {
            const auto& u4_zp = pattern_map.at(u4_zero_point_m);
            zero_point = ov::as_type_ptr<ov::op::v0::Constant>(u4_zp.get_node_shared_ptr());
        }
        if (!weights || !zero_point)
            return false;
        // Due to the matcher specific and Subtract branches similarity,
        // weights and zero_point might be mixed up with each other
        if (ov::shape_size(weights->get_shape()) < ov::shape_size(zero_point->get_shape()))
            std::swap(zero_point, weights);

        const auto& zp_shape = zero_point->get_shape();
        if (ov::shape_size(zp_shape) == 1)
            return false;

        const auto& weights_shape = weights->get_shape();
        // Zero point constant can be converted into scalar only if this does not affect Subtract output shape
        if (weights_shape.size() < zp_shape.size() ||
            !std::equal(zp_shape.rbegin(), zp_shape.rend(), weights_shape.rbegin(), std::less_equal<size_t>())) {
            return false;
        }

        float zp_value;
        if (!ov::op::util::get_single_value(zero_point, zp_value))
            return false;
        const auto new_zp = ov::op::v0::Constant::create(zero_point->get_element_type(), {}, {zp_value});
        return ov::replace_node_update_name(zero_point, new_zp);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(subtract_m, matcher_name);
    register_matcher(m, callback);
}
