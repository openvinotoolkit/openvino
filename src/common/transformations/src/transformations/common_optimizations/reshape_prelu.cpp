// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/reshape_prelu.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::pass;

ReshapePRelu::ReshapePRelu() {
    MATCHER_SCOPE(ReshapePRelu);
    auto input_m = pattern::any_input(pattern::has_static_rank());
    auto slope_m = pattern::any_input(pattern::has_static_rank());
    auto prelu_m = pattern::wrap_type<ov::op::v0::PRelu>({input_m, slope_m});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto prelu = pattern_map.at(prelu_m).get_node_shared_ptr();
        const auto input = pattern_map.at(input_m);
        const auto slope = pattern_map.at(slope_m);

        const auto prelu_pshape = prelu->get_input_partial_shape(0);
        const auto prelu_rank = prelu_pshape.rank();
        const auto slope_pshape = prelu->get_input_partial_shape(1);
        const auto slope_rank = slope_pshape.rank();
        if (prelu_rank.get_length() == 1 || slope_rank.get_length() != 1) {
            return false;
        }

        const auto channel_dim_idx = 1;
        if (slope_pshape.is_static()) {
            const auto slope_shape = slope_pshape.to_shape();
            if (!prelu_pshape[channel_dim_idx].is_dynamic() &&
                static_cast<int64_t>(slope_shape[0]) != prelu_pshape[channel_dim_idx].get_length()) {
                return false;
            }
        }

        std::vector<std::int64_t> target_shape(prelu_rank.get_length(), 1);
        target_shape[channel_dim_idx] = -1;
        const auto target_shape_const =
            ov::op::v0::Constant::create(ov::element::i64, {target_shape.size()}, target_shape);
        auto new_slope = ov::op::util::make_try_fold<ov::op::v1::Reshape>(slope, target_shape_const, true);
        auto new_prelu = prelu->clone_with_new_inputs({input, new_slope});

        replace_node(prelu, new_prelu);
        new_prelu->set_friendly_name(prelu->get_friendly_name());
        copy_runtime_info(prelu, new_prelu);
        copy_runtime_info(slope.get_node_shared_ptr(), new_slope);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(prelu_m, matcher_name);
    this->register_matcher(m, callback);
}
