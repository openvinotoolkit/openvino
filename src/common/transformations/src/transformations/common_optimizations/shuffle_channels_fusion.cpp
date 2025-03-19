// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/shuffle_channels_fusion.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shuffle_channels.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace {
bool check_shapes(const ov::PartialShape& pshape_input,
                  const ov::PartialShape& pshape_reshape_before,
                  const ov::AxisVector& transpose_constant_values,
                  const ov::PartialShape& pshape_reshape_after) {
    // x: [N, C, H, W]
    const auto rank = pshape_input.rank();
    if (rank.is_dynamic() || rank.get_length() != 4) {
        return false;
    }

    // check that all dimensions except batch are static
    if (std::any_of(pshape_input.begin() + 1, pshape_input.end(), [](const ov::Dimension& x) {
            return x.is_dynamic();
        })) {
        return false;
    }

    // x'= reshape(x, [N, group, C / group, H * W]) or reshape(x, [N, group, C / group, H, W])
    if (pshape_reshape_before.rank().get_length() != 4 && pshape_reshape_before.rank().get_length() != 5) {
        return false;
    }

    const auto group = pshape_reshape_before[1].get_length();
    ov::PartialShape expected_reshape_before;
    if (pshape_reshape_before.rank().get_length() == 4) {
        expected_reshape_before = {pshape_input[0],
                                   group,
                                   pshape_input[1].get_length() / group,
                                   pshape_input[2].get_length() * pshape_input[3].get_length()};
    } else {
        expected_reshape_before = {pshape_input[0],
                                   group,
                                   pshape_input[1].get_length() / group,
                                   pshape_input[2],
                                   pshape_input[3]};
    }

    if (!ov::op::util::shapes_equal_except_dynamic_expected_batch(expected_reshape_before, pshape_reshape_before)) {
        return false;
    }

    // x''= transpose(x', [0, 2, 1, 3]) or transpose(x', [0, 2, 1, 3, 4])
    if (transpose_constant_values.size() != 4 && transpose_constant_values.size() != 5) {
        return false;
    }

    ov::AxisVector expected_transpose_values{0, 2, 1, 3};
    if (transpose_constant_values.size() == 5) {
        expected_transpose_values.push_back(4);
    }

    if (expected_transpose_values != transpose_constant_values) {
        return false;
    }

    // y = reshape(x'', [N, C, H, W])
    if (!ov::op::util::shapes_equal_except_dynamic_expected_batch(pshape_input, pshape_reshape_after)) {
        return false;
    }

    return true;
}

}  // namespace

ov::pass::ShuffleChannelsFusion::ShuffleChannelsFusion(const bool reshape_constants_check) {
    MATCHER_SCOPE(ShuffleChannelsFusion);
    auto input = pass::pattern::any_input(pattern::rank_equals(4));
    auto reshape_before_const_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto transpose_const_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto reshape_after_const_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();

    auto reshape_before_pattern =
        ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({input, reshape_before_const_pattern},
                                                          pattern::consumers_count(1));
    auto transpose_pattern =
        ov::pass::pattern::wrap_type<ov::op::v1::Transpose>({reshape_before_pattern, transpose_const_pattern},
                                                            pattern::consumers_count(1));
    auto reshape_after_pattern =
        ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({transpose_pattern, reshape_after_const_pattern});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto data = pattern_map.at(input);
        auto reshape_before_constant =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(reshape_before_const_pattern).get_node_shared_ptr());
        auto reshape_before =
            ov::as_type_ptr<ov::op::v1::Reshape>(pattern_map.at(reshape_before_pattern).get_node_shared_ptr());
        auto transpose =
            ov::as_type_ptr<ov::op::v1::Transpose>(pattern_map.at(transpose_pattern).get_node_shared_ptr());
        auto reshape_after =
            ov::as_type_ptr<ov::op::v1::Reshape>(pattern_map.at(reshape_after_pattern).get_node_shared_ptr());
        auto reshape_after_constant =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(reshape_after_const_pattern).get_node_shared_ptr());
        if (!reshape_after || !transpose || !reshape_before || !reshape_before_constant || !reshape_after_constant) {
            return false;
        }

        if (reshape_constants_check) {
            const auto& reshape_before_values = reshape_before_constant->cast_vector<int64_t>();
            const auto& reshape_after_values = reshape_after_constant->cast_vector<int64_t>();
            if (std::any_of(reshape_before_values.cbegin(),
                            reshape_before_values.cend(),
                            [](const int64_t& value) {
                                return value == -1;
                            }) ||
                std::any_of(reshape_after_values.cbegin(), reshape_after_values.cend(), [](const int64_t& value) {
                    return value == -1;
                })) {
                return false;
            }
        }

        auto pshape_input = reshape_before->get_input_partial_shape(0);
        auto pshape_reshape_before = reshape_before->get_output_partial_shape(0);
        auto pshape_reshape_after = reshape_after->get_output_partial_shape(0);

        auto transpose_constant =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(transpose_const_pattern).get_node_shared_ptr());
        auto transpose_constant_values = transpose_constant->get_axis_vector_val();
        if (!check_shapes(pshape_input, pshape_reshape_before, transpose_constant_values, pshape_reshape_after)) {
            return false;
        }

        int64_t axis = 1ul;
        int64_t group = pshape_reshape_before[1].get_length();

        auto shuffle_shannels = std::make_shared<ov::op::v0::ShuffleChannels>(data, axis, group);
        shuffle_shannels->set_friendly_name(reshape_after->get_friendly_name());
        ov::copy_runtime_info({reshape_before,
                               reshape_before_constant,
                               transpose,
                               transpose_constant,
                               reshape_after,
                               reshape_after_constant},
                              shuffle_shannels);
        ov::replace_node(reshape_after, shuffle_shannels);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reshape_after_pattern, matcher_name);
    register_matcher(m, callback);
}
