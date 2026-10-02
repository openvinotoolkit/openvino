// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/restore_traced_batch.hpp"

#include <algorithm>
#include <memory>

#include "itt.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/roll.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v7 = ov::op::v7;

namespace {

bool keeps_leading_axis(const ov::Output<ov::Node>& order) {
    const auto values = ov::as_type<v0::Constant>(order.get_node())->cast_vector<int64_t>();
    return !values.empty() && values.front() == 0;
}

// Axes of a Roll over the rank-4 merged windows.
bool excludes_leading_axis(const ov::Output<ov::Node>& axes) {
    const auto values = ov::as_type<v0::Constant>(axes.get_node())->cast_vector<int64_t>();
    return std::none_of(values.begin(), values.end(), [](int64_t axis) {
        return axis == 0 || axis == -4;
    });
}

}  // namespace

ov::pass::RestoreTracedBatch::RestoreTracedBatch() {
    MATCHER_SCOPE(RestoreTracedBatch);

    using ov::pass::pattern::any_input;
    using ov::pass::pattern::attrs_match;
    using ov::pass::pattern::consumers_count;
    using ov::pass::pattern::value_matches;
    using ov::pass::pattern::wrap_type;

    const auto axis_zero = attrs_match({{"axis", 0}});

    // window_reverse: view(B, H / ws, W / ws, ws, ws, -1) with B traced to one.
    const auto window_split_target = wrap_type<v0::Concat>({wrap_type<v0::Constant>(value_matches("1")),
                                                            any_input(),
                                                            any_input(),
                                                            any_input(),
                                                            any_input(),
                                                            wrap_type<v0::Constant>(value_matches("-1"))},
                                                           axis_zero) |
                                     wrap_type<v0::Concat>({wrap_type<v0::Constant>(value_matches("1")),
                                                            any_input(),
                                                            any_input(),
                                                            wrap_type<v0::Constant>(value_matches("?, ?, -1"))},
                                                           axis_zero);
    const auto window_split_reshape = wrap_type<v1::Reshape>({any_input(), window_split_target}, consumers_count(1));
    const auto permute = wrap_type<v1::Transpose>({window_split_reshape, wrap_type<v0::Constant>(keeps_leading_axis)},
                                                  consumers_count(1));

    // window_reverse: view(B, H, W, -1) with B traced to one.
    const auto window_merge_target = wrap_type<v0::Concat>({wrap_type<v0::Constant>(value_matches("1")),
                                                            any_input(),
                                                            any_input(),
                                                            wrap_type<v0::Constant>(value_matches("-1"))},
                                                           axis_zero) |
                                     wrap_type<v0::Concat>({wrap_type<v0::Constant>(value_matches("1")),
                                                            wrap_type<v0::Constant>(value_matches("?, ?, -1"))},
                                                           axis_zero);
    const auto window_merge_reshape = wrap_type<v1::Reshape>({permute, window_merge_target}, consumers_count(1));

    // Reverse cyclic shift of shifted windows.
    const auto shift = ov::pass::pattern::optional<v7::Roll>(
        {window_merge_reshape, any_input(), wrap_type<v0::Constant>(excludes_leading_axis)});

    // view(B, H * W, C) taking B from the input shape.
    const auto input_batch = ov::pass::pattern::optional<v0::Convert>(
        {wrap_type<ov::op::util::GatherBase>({wrap_type<v0::ShapeOf, v3::ShapeOf>({wrap_type<v0::Parameter>()}),
                                              wrap_type<v0::Constant>(value_matches("0")),
                                              any_input()})});
    const auto rebuild_target = wrap_type<v0::Concat>({input_batch, any_input(), any_input()}, axis_zero) |
                                wrap_type<v0::Concat>({input_batch, any_input()}, axis_zero);
    const auto rebuild = wrap_type<v1::Reshape>({shift, rebuild_target});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& matcher) {
        const auto& pattern_map = matcher.get_pattern_value_map();
        const auto batch = pattern_map.at(rebuild_target).get_node()->input_value(0);
        const auto window_split_target_node = pattern_map.at(window_split_target).get_node_shared_ptr();
        const auto window_merge_target_node = pattern_map.at(window_merge_target).get_node_shared_ptr();
        if (window_split_target_node->get_element_type() != batch.get_element_type() ||
            window_merge_target_node->get_element_type() != batch.get_element_type()) {
            return false;
        }

        window_split_target_node->input(0).replace_source_output(batch);
        window_merge_target_node->input(0).replace_source_output(batch);
        return true;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(rebuild, matcher_name), callback);
}
