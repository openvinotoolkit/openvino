// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/restore_traced_batch.hpp"

#include <algorithm>
#include <memory>

#include "itt.hpp"
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
    const auto p_window_split_target = wrap_type<v0::Concat>({wrap_type<v0::Constant>(value_matches("1")),
                                                              any_input(),
                                                              any_input(),
                                                              any_input(),
                                                              any_input(),
                                                              wrap_type<v0::Constant>(value_matches("-1"))},
                                                             axis_zero);
    const auto p_window_split_reshape =
        wrap_type<v1::Reshape>({any_input(), p_window_split_target}, consumers_count(1));
    const auto p_permute =
        wrap_type<v1::Transpose>({p_window_split_reshape, wrap_type<v0::Constant>(transpose_keeps_batch_leading)},
                                 consumers_count(1));

    // window_reverse: view(B, H, W, -1) with B traced to one.
    const auto p_window_merge_target = wrap_type<v0::Concat>({wrap_type<v0::Constant>(value_matches("1")),
                                                              any_input(),
                                                              any_input(),
                                                              wrap_type<v0::Constant>(value_matches("-1"))},
                                                             axis_zero);
    const auto p_window_merge_reshape = wrap_type<v1::Reshape>({p_permute, p_window_merge_target}, consumers_count(1));

    // Reverse cyclic shift of shifted windows.
    const auto p_shift =
        ov::pass::pattern::optional<v7::Roll>({p_window_merge_reshape, any_input(), wrap_type<v0::Constant>()},
                                              consumers_count(1) && roll_skips_batch);

    // view(B, H * W, C) taking B from the input shape.
    const auto p_input_batch = ov::pass::pattern::optional<v0::Convert>(
        {wrap_type<ov::op::util::GatherBase>({wrap_type<ov::op::util::ShapeOfBase>({wrap_type<v0::Parameter>()}),
                                              wrap_type<v0::Constant>(value_matches("0")),
                                              any_input()})});
    const auto p_rebuild_target = wrap_type<v0::Concat>({p_input_batch, any_input(), any_input()}, axis_zero);
    const auto p_rebuild = wrap_type<v1::Reshape>({p_shift, p_rebuild_target});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& matcher) {
        const auto& pattern_map = matcher.get_pattern_value_map();
        // An absent optional Convert leaves no record, so the batch is read from the matched target.
        const auto batch = pattern_map.at(p_rebuild_target).get_node()->input_value(0);
        const auto split_target = pattern_map.at(p_window_split_target);
        const auto merge_target = pattern_map.at(p_window_merge_target);
        if (split_target.get_element_type() != batch.get_element_type() ||
            merge_target.get_element_type() != batch.get_element_type()) {
            return false;
        }

        split_target.get_node()->input(0).replace_source_output(batch);
        merge_target.get_node()->input(0).replace_source_output(batch);
        return true;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(p_rebuild, matcher_name), callback);
}
