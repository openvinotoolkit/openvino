// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/softmax_fusion.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::SoftmaxFusion::SoftmaxFusion() {
    MATCHER_SCOPE(SoftmaxFusion);

    auto data_pattern = pass::pattern::any_input(pass::pattern::has_static_rank());
    auto reduce_max_axes_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto reduce_max_pattern =
        ov::pass::pattern::wrap_type<ov::op::v1::ReduceMax>({data_pattern, reduce_max_axes_pattern});
    auto sub_pattern = ov::pass::pattern::wrap_type<ov::op::v1::Subtract>({data_pattern, reduce_max_pattern});

    auto exp_input = std::make_shared<pattern::op::Or>(OutputVector{sub_pattern, data_pattern});
    auto exp_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Exp>({exp_input});

    auto reduce_sum_axes_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto reduce_sum_pattern =
        ov::pass::pattern::wrap_type<ov::op::v1::ReduceSum>({exp_pattern, reduce_sum_axes_pattern});
    auto div_pattern = ov::pass::pattern::wrap_type<ov::op::v1::Divide>({exp_pattern, reduce_sum_pattern});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root()))
            return false;

        const auto& pattern_map = m.get_pattern_value_map();

        auto reduce_sum_axes =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(reduce_sum_axes_pattern).get_node_shared_ptr());
        if (!reduce_sum_axes || shape_size(reduce_sum_axes->get_shape()) != 1)
            return false;
        int64_t reduce_sum_axis = reduce_sum_axes->cast_vector<int64_t>()[0];
        if (reduce_sum_axis < 0) {
            const auto& rank = pattern_map.at(data_pattern).get_partial_shape().rank().get_length();
            reduce_sum_axis += rank;
        }

        auto exp_input_is_subtract = pattern_map.count(sub_pattern) != 0;
        if (exp_input_is_subtract) {
            auto reduce_max_axes =
                ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(reduce_max_axes_pattern).get_node_shared_ptr());
            if (!reduce_max_axes || shape_size(reduce_max_axes->get_shape()) != 1)
                return false;
            int64_t reduce_max_axis = reduce_max_axes->cast_vector<int64_t>()[0];

            if (reduce_max_axis < 0) {
                const auto& rank = pattern_map.at(data_pattern).get_partial_shape().rank().get_length();
                reduce_max_axis += rank;
            }

            if (reduce_max_axis != reduce_sum_axis)
                return false;
        }

        auto softmax = register_new_node<ov::op::v1::Softmax>(pattern_map.at(data_pattern), reduce_sum_axis);
        auto div = pattern_map.at(div_pattern).get_node_shared_ptr();
        softmax->set_friendly_name(div->get_friendly_name());

        NodeVector fused_nodes;
        fused_nodes.push_back(pattern_map.at(exp_pattern).get_node_shared_ptr());
        fused_nodes.push_back(pattern_map.at(reduce_sum_pattern).get_node_shared_ptr());
        fused_nodes.push_back(div);
        if (exp_input_is_subtract) {
            fused_nodes.push_back(pattern_map.at(reduce_max_pattern).get_node_shared_ptr());
            fused_nodes.push_back(pattern_map.at(sub_pattern).get_node_shared_ptr());
        }
        copy_runtime_info(fused_nodes, softmax);

        replace_node(div, softmax);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(div_pattern, matcher_name);
    this->register_matcher(m, callback);
}
