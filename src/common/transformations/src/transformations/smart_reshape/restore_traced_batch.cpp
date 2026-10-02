// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/restore_traced_batch.hpp"

#include <algorithm>
#include <memory>
#include <utility>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/roll.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v7 = ov::op::v7;

namespace {

// The restored batch must stay the leading axis; otherwise it was not the traced batch.
bool transpose_keeps_batch_leading(const ov::Output<ov::Node>& order) {
    const auto values = ov::as_type<v0::Constant>(order.get_node())->cast_vector<int64_t>();
    return !values.empty() && values.front() == 0;
}

// Rolling over the traced batch of one is a no-op, but over a restored batch it would mix samples.
bool roll_skips_batch(const ov::Output<ov::Node>& roll) {
    const auto rank = roll.get_partial_shape().rank();
    const auto axes = ov::as_type<v0::Constant>(roll.get_node()->get_input_node_ptr(2));
    if (rank.is_dynamic() || !axes) {
        return false;
    }
    const auto values = axes->cast_vector<int64_t>();
    return std::none_of(values.begin(), values.end(), [&](int64_t axis) {
        return ov::util::normalize_axis(axis, rank.get_length()) == 0;
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
    const auto permute =
        wrap_type<v1::Transpose>({window_split_reshape, wrap_type<v0::Constant>(transpose_keeps_batch_leading)},
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
    const auto shift =
        ov::pass::pattern::optional<v7::Roll>({window_merge_reshape, any_input(), wrap_type<v0::Constant>()},
                                              consumers_count(1) && roll_skips_batch);

    // view(B, H * W, C) taking B from the input shape.
    const auto input_batch = ov::pass::pattern::optional<v0::Convert>(
        {wrap_type<ov::op::util::GatherBase>({wrap_type<ov::op::util::ShapeOfBase>({wrap_type<v0::Parameter>()}),
                                              wrap_type<v0::Constant>(value_matches("0")),
                                              any_input()})});
    const auto rebuild_target = wrap_type<v0::Concat>({input_batch, any_input(), any_input()}, axis_zero) |
                                wrap_type<v0::Concat>({input_batch, any_input()}, axis_zero);
    const auto rebuild = wrap_type<v1::Reshape>({shift, rebuild_target});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& matcher) {
        const auto& pattern_map = matcher.get_pattern_value_map();
        const auto batch = pattern_map.at(rebuild_target).get_node()->input_value(0);
        const std::pair<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> pinned_reshapes[] = {
            {window_split_reshape, window_split_target},
            {window_merge_reshape, window_merge_target}};
        for (const auto& [reshape, target] : pinned_reshapes) {
            if (pattern_map.at(target).get_element_type() != batch.get_element_type()) {
                return false;
            }
        }

        for (const auto& [reshape, target] : pinned_reshapes) {
            const auto pinned_target = pattern_map.at(target).get_node_shared_ptr();
            if (pinned_target->get_output_target_inputs(0).size() == 1) {
                pinned_target->input(0).replace_source_output(batch);
                continue;
            }
            // Shared with other Reshapes, so only the matched one gets a restored copy.
            auto inputs = pinned_target->input_values();
            inputs[0] = batch;
            const auto restored_target = pinned_target->clone_with_new_inputs(inputs);
            ov::copy_runtime_info(pinned_target, restored_target);
            pattern_map.at(reshape).get_node()->input(1).replace_source_output(restored_target);
        }
        return true;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(rebuild, matcher_name), callback);
}
