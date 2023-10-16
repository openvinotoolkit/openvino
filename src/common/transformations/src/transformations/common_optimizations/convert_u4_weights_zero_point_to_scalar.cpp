// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_u4_weights_zero_point_to_scalar.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/reference/autobroadcast_binop.hpp"
#include "transformations/utils/utils.hpp"

namespace {
bool convert_zero_point_to_scalar(const std::shared_ptr<ov::op::v0::Constant>& weights,
                                  const std::shared_ptr<ov::op::v0::Constant>& zero_point) {
    auto zero_point_shape = zero_point->get_shape();
    if (ov::shape_size(zero_point_shape) == 1)
        return false;

    const auto& weights_shape = weights->get_shape();
    const size_t weights_rank = weights_shape.size();
    const size_t zero_point_rank = zero_point_shape.size();
    // Zero point constant can be converted into scalar only if this does not affect Subtract output shape
    if (weights_rank < zero_point_rank || ov::shape_size(weights_shape) < ov::shape_size(zero_point_shape))
        return false;

    zero_point_shape.insert(zero_point_shape.begin(), weights_rank - zero_point_rank, 1);
    for (size_t i = 0; i < weights_rank; ++i) {
        if (zero_point_shape[i] > weights_shape[i])
            return false;
    }

    int8_t zero_point_value = zero_point->cast_vector<int8_t>(1)[0];
    if (!ov::op::util::constantIsEqualTo(zero_point, static_cast<float>(zero_point_value)))
        return false;

    const auto new_zp = ov::op::v0::Constant::create(zero_point->get_element_type(), {}, {zero_point_value});
    return ov::replace_node_update_name(zero_point, new_zp);
}
}  // namespace

ov::pass::ConvertU4WeightsFloatZeroPointToScalar::ConvertU4WeightsFloatZeroPointToScalar() {
    MATCHER_SCOPE(ConvertU4WeightsFloatZeroPointToScalar);
    auto weights_m = pattern::wrap_type<ov::op::v0::Constant>(pattern::type_matches(ov::element::u4));
    auto convert_m = pattern::wrap_type<ov::op::v0::Convert>({weights_m}, pattern::consumers_count(1));
    auto zero_point_m = pattern::wrap_type<ov::op::v0::Constant>(pattern::consumers_count(1));
    auto subtract_m = pattern::wrap_type<ov::op::v1::Subtract>({convert_m, zero_point_m});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto& pattern_map = m.get_pattern_value_map();
        auto weights = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(weights_m).get_node_shared_ptr());
        auto zero_point = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(zero_point_m).get_node_shared_ptr());
        if (!weights || !zero_point)
            return false;
        return convert_zero_point_to_scalar(weights, zero_point);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(subtract_m, matcher_name);
    register_matcher(m, callback);
}

ov::pass::ConvertU4WeightsU4ZeroPointToScalar::ConvertU4WeightsU4ZeroPointToScalar() {
    MATCHER_SCOPE(ConvertU4WeightsU4ZeroPointToScalar);
    auto weights_m = pattern::wrap_type<ov::op::v0::Constant>(pattern::type_matches(ov::element::u4));
    auto weights_convert_m = pattern::wrap_type<ov::op::v0::Convert>({weights_m});
    auto zp_predicate = [](ov::Output<ov::Node> output) -> bool {
        return pattern::type_matches(ov::element::u4)(output) && pattern::consumers_count(1)(output);
    };
    auto zero_point_m = pattern::wrap_type<ov::op::v0::Constant>(zp_predicate);
    auto zero_point_convert_m = pattern::wrap_type<ov::op::v0::Convert>({zero_point_m}, pattern::consumers_count(1));
    auto subtract_m = pattern::wrap_type<ov::op::v1::Subtract>({weights_convert_m, zero_point_convert_m});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto& pattern_map = m.get_pattern_value_map();
        auto weights = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(weights_m).get_node_shared_ptr());
        auto zero_point = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(zero_point_m).get_node_shared_ptr());
        if (!weights || !zero_point)
            return false;
        // Due to the matcher specific and Subtract branches similarity,
        // weights and zero_point might be mixed up with each other
        if (ov::shape_size(weights->get_shape()) < ov::shape_size(zero_point->get_shape())) {
            std::swap(zero_point, weights);
        }
        return convert_zero_point_to_scalar(weights, zero_point);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(subtract_m, matcher_name);
    register_matcher(m, callback);
}

ov::pass::ConvertU4WeightsZeroPointToScalar::ConvertU4WeightsZeroPointToScalar() {
    this->add_matcher<ConvertU4WeightsFloatZeroPointToScalar>();
    this->add_matcher<ConvertU4WeightsU4ZeroPointToScalar>();
}
