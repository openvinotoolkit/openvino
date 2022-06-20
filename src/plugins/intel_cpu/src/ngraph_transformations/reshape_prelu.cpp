// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_prelu.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "transformations/utils/utils.hpp"

#include "itt.hpp"

ov::intel_cpu::ReshapePRelu::ReshapePRelu() {
    MATCHER_SCOPE(ReshapeFullyConnectedFusion);
    auto input_m = ngraph::pattern::any_input(ngraph::pattern::has_static_rank());
    auto slope_m = ngraph::pattern::any_input(ngraph::pattern::has_static_rank());
    auto prelu_m = ngraph::pattern::wrap_type<ngraph::opset1::PRelu>({ input_m, slope_m });

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
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
            if (!prelu_pshape[channel_dim_idx].is_dynamic() && slope_shape[0] != prelu_pshape[channel_dim_idx].get_length()) {
                return false;
            }
        }

        std::vector<std::int64_t> target_shape(prelu_rank.get_length(), 1);
        target_shape[channel_dim_idx] = -1;
        const auto target_shape_const = ngraph::opset1::Constant::create(ngraph::element::i64, { target_shape.size() }, target_shape);
        auto new_slope = ngraph::op::util::make_try_fold<ngraph::opset1::Reshape>(slope, target_shape_const, true);
        auto new_prelu = prelu->clone_with_new_inputs({ input, new_slope });

        ngraph::replace_node(prelu, new_prelu);
        new_prelu->set_friendly_name(prelu->get_friendly_name());
        ngraph::copy_runtime_info(prelu, new_prelu);
        MATCHER_SCOPE_ENABLE(ReshapeFullyConnectedFusion);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(prelu_m, matcher_name);
    this->register_matcher(m, callback);
}
