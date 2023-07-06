// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/group_normalization_decomposition.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::pass;
using namespace ov::opset12;

namespace {
// This function creates a shape to which we need to reshape the input
// before normalization.
// If data shape is [N,C,H,W], the function returns
// [N * num_groups, C // num_groups, H, W]
std::shared_ptr<Node> create_group_norm_shape(NodeRegistry& reg, const Output<Node>& data, size_t num_groups) {
    const auto& pshape = data.get_partial_shape();
    NGRAPH_CHECK(pshape.rank().is_static());
    size_t rank_size = pshape.rank().get_length();
    NGRAPH_CHECK(rank_size >= 3, "3-D and above tensors supported only");

    auto shape = reg.make<ShapeOf>(data);

    const auto axis_node = reg.add(Constant::create(element::i64, Shape{}, {0}));
    const auto split = reg.make<Split>(shape, axis_node, rank_size);
    auto splits = split->outputs();
    auto num_groups_const = reg.add(Constant::create(element::i64, Shape{1}, {num_groups}));
    // The 4D shape: [N * num_groups, C // num_groups, H, W] is created
    // instead of 5D shape: [N, num_groups, C // num_groups, H, W].
    // The reason is the lack of support for 5D MVN input by some plugins.
    OutputVector new_shape{reg.make<Multiply>(splits[0], num_groups_const),
                           reg.make<Divide>(splits[1], num_groups_const)};

    for (size_t i = 2; i < rank_size; i++) {
        new_shape.push_back(splits[i]);
    }
    return reg.make<Concat>(new_shape, 0);
}

Output<Node> reshape_channel_shaped_node_to_nchw(NodeRegistry& reg,
                                                 const Output<Node>& node,
                                                 const Output<Node>& expected_rank) {
    const auto one_const = reg.add(Constant::create(element::i64, Shape{1}, {1}));
    const auto two_const = reg.add(Constant::create(element::i64, Shape{1}, {2}));
    const auto tail_shape_rank = reg.make<Subtract>(expected_rank, two_const);
    const auto tail_shape = reg.make<Broadcast>(one_const, tail_shape_rank);

    // Construct new bias shape: [1, C, 1, 1, ... ]
    const auto C_dim = reg.make<ShapeOf>(node);
    const auto new_shape = reg.make<Concat>(OutputVector{one_const, C_dim, tail_shape}, 0);

    return reg.make<Reshape>(node, new_shape, false);
}

template <typename T>
std::vector<T> get_monotonic_range(T end_value, T start_value = T{0}, T step = T{1}) {
    auto value_count = static_cast<std::size_t>(std::floor((end_value - start_value) / step));

    std::vector<T> range(value_count);

    // Calculate initial value (one step below starting value)
    size_t n = start_value - step;
    // Generate a vector of values by adding step to previous value
    std::generate(std::begin(range), std::end(range), [&n, &step]() -> T {
        return n += step;
    });

    return range;
}

std::shared_ptr<Node> get_monotonic_range_along_node_rank(NodeRegistry& reg,
                                                          const Output<Node>& value,
                                                          int64_t start_value,
                                                          int64_t step) {
    if (value.get_partial_shape().rank().is_static()) {
        const auto range_value =
            get_monotonic_range<int64_t>(value.get_partial_shape().rank().get_length(), start_value, step);
        return reg.add(Constant::create(element::i64, {range_value.size()}, range_value));
    }

    const auto value_shape = reg.make<ShapeOf>(value);
    return reg.make<Range>(reg.add(Constant::create(element::i64, {}, {start_value})),
                           reg.make<ShapeOf>(value_shape),
                           reg.add(Constant::create(element::i64, {}, {step})),
                           element::i64

    );
}

}  // namespace

ov::pass::GroupNormalizationDecomposition::GroupNormalizationDecomposition() {
    MATCHER_SCOPE(GroupNormalizationDecomposition);

    auto group_norm_pattern = pattern::wrap_type<GroupNormalization>();

    matcher_pass_callback callback = [=](pattern::Matcher& matcher) {
        NodeRegistry reg;

        auto group_norm_node = std::dynamic_pointer_cast<GroupNormalization>(matcher.get_match_root());
        if (!group_norm_node || transformation_callback(group_norm_node) || group_norm_node->get_input_size() != 3) {
            return false;
        }

        auto data = group_norm_node->input_value(0);
        auto scale = group_norm_node->input_value(1);
        auto bias = group_norm_node->input_value(2);

        size_t num_groups =
            static_cast<size_t>(group_norm_node->get_num_groups());  // Negative values are checked by op validation
        float eps = group_norm_node->get_epsilon();

        auto data_shape_node = reg.make<ShapeOf>(data);
        auto data_reshaped = reg.make<Reshape>(data, create_group_norm_shape(reg, data, num_groups), true);
        const auto reduction_axes = get_monotonic_range_along_node_rank(reg, data_reshaped, 1, 1);

        auto mvn = reg.make<MVN>(data_reshaped, reduction_axes, true, eps, op::MVNEpsMode::INSIDE_SQRT);
        std::shared_ptr<Node> result = reg.make<Reshape>(mvn, data_shape_node, true);

        const auto& scale_shape = scale.get_partial_shape();
        const auto& bias_shape = bias.get_partial_shape();

        if (!scale_shape.rank().is_static() || !bias_shape.rank().is_static()) {
            return false;
        }

        const auto data_rank = reg.make<ShapeOf>(data_shape_node);
        result = reg.make<Multiply>(result, reshape_channel_shaped_node_to_nchw(reg, scale, data_rank));
        result = reg.make<Add>(result, reshape_channel_shaped_node_to_nchw(reg, bias, data_rank));

        copy_runtime_info(group_norm_node, reg.get());
        replace_node(group_norm_node, result);

        return true;
    };

    auto m = make_shared<pattern::Matcher>(group_norm_pattern, matcher_name);
    this->register_matcher(m, callback);
}
