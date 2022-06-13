// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/softmax_fusion.hpp"

#include <memory>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <vector>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

ngraph::pass::SoftmaxFusion::SoftmaxFusion() {
    MATCHER_SCOPE(SoftmaxFusion);
    auto data_pattern = ngraph::pattern::any_input(pattern::has_static_rank());
    auto reduce_max_axes_pattern = ngraph::pattern::wrap_type<opset6::Constant>();
    auto reduce_max_pattern = ngraph::pattern::wrap_type<opset6::ReduceMax>({data_pattern, reduce_max_axes_pattern});
    auto sub_pattern = ngraph::pattern::wrap_type<opset6::Subtract>({data_pattern, reduce_max_pattern});
    auto exp_pattern = ngraph::pattern::wrap_type<opset6::Exp>({sub_pattern});
    auto reduce_sum_axes_pattern = ngraph::pattern::wrap_type<opset6::Constant>();
    auto reduce_sum_pattern = ngraph::pattern::wrap_type<opset6::ReduceSum>({exp_pattern, reduce_sum_axes_pattern});
    auto div_pattern = ngraph::pattern::wrap_type<opset6::Divide>({exp_pattern, reduce_sum_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root()))
            return false;

        const auto& pattern_map = m.get_pattern_value_map();

        auto reduce_max_axes =
            std::dynamic_pointer_cast<opset6::Constant>(pattern_map.at(reduce_max_axes_pattern).get_node_shared_ptr());
        if (!reduce_max_axes || shape_size(reduce_max_axes->get_shape()) != 1)
            return false;
        auto reduce_sum_axes =
            std::dynamic_pointer_cast<opset6::Constant>(pattern_map.at(reduce_sum_axes_pattern).get_node_shared_ptr());
        if (!reduce_sum_axes || shape_size(reduce_sum_axes->get_shape()) != 1)
            return false;

        int64_t reduce_max_axis = reduce_max_axes->cast_vector<int64_t>()[0];
        int64_t reduce_sum_axis = reduce_sum_axes->cast_vector<int64_t>()[0];
        if (reduce_max_axis < 0 || reduce_sum_axis < 0) {
            const auto& pshape = pattern_map.at(data_pattern).get_partial_shape();
            auto rank = pshape.rank().get_length();
            if (reduce_max_axis < 0)
                reduce_max_axis += rank;
            if (reduce_sum_axis < 0)
                reduce_sum_axis += rank;
        }
        if (reduce_max_axis != reduce_sum_axis)
            return false;

        auto softmax = register_new_node<ngraph::opset6::Softmax>(pattern_map.at(data_pattern), reduce_sum_axis);
        auto div = pattern_map.at(div_pattern).get_node_shared_ptr();
        softmax->set_friendly_name(div->get_friendly_name());

        copy_runtime_info(
            {
                pattern_map.at(reduce_max_pattern).get_node_shared_ptr(),
                pattern_map.at(sub_pattern).get_node_shared_ptr(),
                pattern_map.at(exp_pattern).get_node_shared_ptr(),
                pattern_map.at(reduce_sum_pattern).get_node_shared_ptr(),
                div,
            },
            softmax);
        replace_node(div, softmax);
        MATCHER_SCOPE_ENABLE(SoftmaxFusion);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(div_pattern, matcher_name);
    this->register_matcher(m, callback);
}
