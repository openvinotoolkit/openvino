// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/group_normalization_decomposition.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace ov;
using namespace op;
using namespace ov::pass;

namespace {
// This function creates a shape to which we need to reshape the input
// before normalization.
// If data shape is [N,C,H,W], the function returns
// [N * num_groups, C // num_groups, H, W]
std::shared_ptr<Node> create_group_norm_shape(NodeRegistry& reg,
                                              const Output<Node>& shape,
                                              size_t num_groups,
                                              size_t rank_size) {
    const auto axis_node = reg.add(v0::Constant::create(element::i64, Shape{}, {0}));
    const auto split = reg.make<v1::Split>(shape, axis_node, rank_size);
    auto splits = split->outputs();
    auto num_groups_const = reg.add(v0::Constant::create(element::i64, Shape{1}, {num_groups}));
    // The 4D shape: [N * num_groups, C // num_groups, H, W] is created
    // instead of 5D shape: [N, num_groups, C // num_groups, H, W].
    // The reason is the lack of support for 5D MVN input by some plugins.
    OutputVector new_shape{reg.make<v1::Multiply>(splits[0], num_groups_const),
                           reg.make<v1::Divide>(splits[1], num_groups_const)};

    std::move(splits.begin() + 2, splits.end(), std::back_inserter(new_shape));
    return reg.make<v0::Concat>(new_shape, 0);
}

std::shared_ptr<Node> get_range(NodeRegistry& reg, int64_t start, int64_t stop) {
    std::vector<int64_t> range_values(stop - start);
    std::iota(range_values.begin(), range_values.end(), start);
    return reg.add(v0::Constant::create(element::i64, {range_values.size()}, range_values));
}
}  // namespace

ov::pass::GroupNormalizationDecomposition::GroupNormalizationDecomposition() {
    MATCHER_SCOPE(GroupNormalizationDecomposition);

    auto group_norm_pattern = pattern::wrap_type<v12::GroupNormalization>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& matcher) {
        NodeRegistry reg;

        const auto group_norm_node = ov::as_type_ptr<v12::GroupNormalization>(matcher.get_match_root());
        if (!group_norm_node || transformation_callback(group_norm_node) || group_norm_node->get_input_size() != 3) {
            return false;
        }

        const auto data = group_norm_node->input_value(0);
        const auto scale = group_norm_node->input_value(1);
        const auto bias = group_norm_node->input_value(2);

        const auto num_groups =
            static_cast<size_t>(group_norm_node->get_num_groups());  // Negative values are checked by op validation
        const auto eps = op::util::cast_eps_to_float(group_norm_node->get_epsilon());

        const auto& data_rank = data.get_partial_shape().rank();
        if (data_rank.is_dynamic() || data_rank.get_length() < 3) {
            return false;
        }
        const auto data_rank_size = data_rank.get_length();
        const auto data_shape_node = reg.make<v3::ShapeOf>(data);
        const auto data_reshaped = reg.make<v1::Reshape>(
            data,
            create_group_norm_shape(reg, data_shape_node, num_groups, static_cast<size_t>(data_rank_size)),
            true);
        const auto reduction_axes = get_range(reg, 1, data_rank_size);

        const auto mvn = reg.make<v6::MVN>(data_reshaped, reduction_axes, true, eps, op::MVNEpsMode::INSIDE_SQRT);
        std::shared_ptr<Node> result = reg.make<v1::Reshape>(mvn, data_shape_node, true);

        // Unsqueeze scale and bias to shape: [C, 1, 1, ... ]
        const auto unsqueeze_axes = get_range(reg, 1, data_rank_size - 1);
        result = reg.make<v1::Multiply>(result, reg.make<v0::Unsqueeze>(scale, unsqueeze_axes));
        result = reg.make<v1::Add>(result, reg.make<v0::Unsqueeze>(bias, unsqueeze_axes));

        result->set_friendly_name(group_norm_node->get_friendly_name());

        copy_runtime_info(group_norm_node, reg.get());
        replace_node(group_norm_node, result);

        return true;
    };

    auto m = make_shared<pattern::Matcher>(group_norm_pattern, matcher_name);
    this->register_matcher(m, callback);
}
