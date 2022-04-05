// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/batch_to_space_fusion.hpp"

#include <memory>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <vector>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

ngraph::pass::BatchToSpaceFusion::BatchToSpaceFusion() {
    MATCHER_SCOPE(BatchToSpaceFusion);
    auto data_pattern = pattern::any_input(pattern::has_static_shape());
    auto reshape_before_pattern =
        pattern::wrap_type<opset6::Reshape>({data_pattern, pattern::wrap_type<opset6::Constant>()},
                                            pattern::rank_equals(4));
    auto trans_before_pattern =
        pattern::wrap_type<opset6::Transpose>({data_pattern, pattern::wrap_type<opset6::Constant>()},
                                              pattern::rank_equals(4));
    auto reshape_or_transpose_before_pattern =
        std::make_shared<pattern::op::Or>(OutputVector{reshape_before_pattern, trans_before_pattern});
    auto depth_to_space_pattern = pattern::wrap_type<opset6::DepthToSpace>({reshape_or_transpose_before_pattern});
    auto starts_pattern = pattern::wrap_type<opset6::Constant>();
    auto ends_pattern = pattern::wrap_type<opset6::Constant>();
    auto slice_pattern = pattern::wrap_type<opset6::StridedSlice>(
        {depth_to_space_pattern, starts_pattern, ends_pattern, pattern::wrap_type<opset6::Constant>()});
    auto reshape_after_pattern =
        pattern::wrap_type<opset6::Reshape>({slice_pattern, pattern::wrap_type<opset6::Constant>()},
                                            pattern::rank_equals(4));
    auto trans_after_pattern =
        pattern::wrap_type<opset6::Transpose>({slice_pattern, pattern::wrap_type<opset6::Constant>()},
                                              pattern::rank_equals(4));
    auto reshape_or_transpose_after_pattern =
        std::make_shared<pattern::op::Or>(OutputVector{reshape_after_pattern, trans_after_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto get_reshape_or_transpose = [&pattern_map](
                                            const std::shared_ptr<Node>& reshape_pattern,
                                            const std::shared_ptr<Node>& trans_pattern) -> std::shared_ptr<Node> {
            if (pattern_map.count(reshape_pattern))
                return pattern_map.at(reshape_pattern).get_node_shared_ptr();
            if (pattern_map.count(trans_pattern))
                return pattern_map.at(trans_pattern).get_node_shared_ptr();
            return nullptr;
        };
        auto check_input_output_shape = [](const std::shared_ptr<Node>& node) -> bool {
            const auto& input_shape = node->get_input_shape(0);
            const auto& output_shape = node->get_output_shape(0);
            // Transpose permutation has to be [1, 0, 2, 3]
            return input_shape[0] == output_shape[1] && input_shape[1] == output_shape[0] &&
                   input_shape[2] == output_shape[2] && input_shape[3] == output_shape[3];
        };

        std::shared_ptr<Node> reshape_or_trans_before =
            get_reshape_or_transpose(reshape_before_pattern, trans_before_pattern);
        if (!reshape_or_trans_before)
            return false;
        if (!check_input_output_shape(reshape_or_trans_before))
            return false;
        std::shared_ptr<Node> reshape_or_trans_after =
            get_reshape_or_transpose(reshape_after_pattern, trans_after_pattern);
        if (!reshape_or_trans_after)
            return false;
        if (!check_input_output_shape(reshape_or_trans_after))
            return false;

        auto depth_to_space = std::dynamic_pointer_cast<opset6::DepthToSpace>(
            pattern_map.at(depth_to_space_pattern).get_node_shared_ptr());
        if (!depth_to_space)
            return false;
        if (depth_to_space->get_mode() != opset6::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST)
            return false;
        const auto& dts_shape = depth_to_space->get_shape();
        if (dts_shape.size() != 4)
            return false;
        auto block_size = static_cast<int64_t>(depth_to_space->get_block_size());
        auto block_shape =
            op::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{1, 1, block_size, block_size});
        auto starts = std::dynamic_pointer_cast<opset6::Constant>(pattern_map.at(starts_pattern).get_node_shared_ptr());
        if (!starts)
            return false;
        auto ends = std::dynamic_pointer_cast<opset6::Constant>(pattern_map.at(ends_pattern).get_node_shared_ptr());
        if (!ends)
            return false;
        auto starts_value = starts->cast_vector<int64_t>();
        auto ends_value = ends->cast_vector<int64_t>();
        // Convert StridedSlice's 'ends' input to BatchToSpace's 'crops_ends'
        for (size_t i = 0; i < ends_value.size(); i++) {
            if (ends_value[i] < 0) {
                // negative ends become positive crops
                // e.g. ends[i] == -2 means cropping i-th dimension by 2 from the back
                ends_value[i] = -ends_value[i];
            } else if (ends_value[i] > static_cast<int64_t>(dts_shape[i])) {
                // no cropping from the back if ends[i] > shape[i]
                ends_value[i] = 0;
            } else {
                // else if ends[i] is positive and within [0, shape[i]] - crop the difference: shape[i] - ends[i]
                ends_value[i] = dts_shape[i] - ends_value[i];
            }
        }
        auto crops_begin = op::Constant::create(element::i64, Shape{4}, starts_value);
        auto crops_end = op::Constant::create(element::i64, Shape{4}, ends_value);
        auto batch_to_space = register_new_node<ngraph::opset6::BatchToSpace>(pattern_map.at(data_pattern),
                                                                              block_shape,
                                                                              crops_begin,
                                                                              crops_end);
        batch_to_space->set_friendly_name(reshape_or_trans_after->get_friendly_name());

        copy_runtime_info({reshape_or_trans_before,
                           depth_to_space,
                           pattern_map.at(slice_pattern).get_node_shared_ptr(),
                           reshape_or_trans_after},
                          batch_to_space);
        replace_node(reshape_or_trans_after, batch_to_space);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape_or_transpose_after_pattern, matcher_name);
    this->register_matcher(m, callback);
}
