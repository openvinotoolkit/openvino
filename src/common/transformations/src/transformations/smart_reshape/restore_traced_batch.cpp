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
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v7 = ov::op::v7;

namespace {

// A shape target [1, ..., -1] of the given length: the batch traced to one and the rest inferred.
auto is_pinned_target(int64_t length) {
    return [length](const ov::Output<ov::Node>& output) {
        const auto concat = ov::as_type<v0::Concat>(output.get_node());
        if (concat->get_axis() != 0 || output.get_partial_shape() != ov::PartialShape{length} ||
            !ov::op::util::has_constant_value<int64_t>(concat->get_input_node_shared_ptr(0), 1)) {
            return false;
        }
        const auto last = ov::as_type<v0::Constant>(concat->get_input_node_ptr(concat->get_input_size() - 1));
        if (!last) {
            return false;
        }
        const auto values = last->cast_vector<int64_t>();
        return !values.empty() && values.back() == -1;
    };
}

// Gather(ShapeOf(Parameter), 0), optionally converted.
bool is_input_batch(const ov::Output<ov::Node>& output) {
    auto node = output.get_node();
    if (ov::is_type<v0::Convert>(node)) {
        node = node->get_input_node_ptr(0);
    }
    const auto gather = ov::as_type<ov::op::util::GatherBase>(node);
    if (!gather || !ov::op::util::has_constant_value<int64_t>(gather->get_input_node_shared_ptr(1), 0)) {
        return false;
    }
    const auto shape_of = gather->get_input_node_ptr(0);
    return (ov::is_type<v0::ShapeOf>(shape_of) || ov::is_type<v3::ShapeOf>(shape_of)) &&
           ov::is_type<v0::Parameter>(shape_of->get_input_node_ptr(0));
}

bool is_batch_rebuild_target(const ov::Output<ov::Node>& output) {
    const auto concat = output.get_node();
    return ov::as_type<v0::Concat>(concat)->get_axis() == 0 && output.get_partial_shape() == ov::PartialShape{3} &&
           is_input_batch(concat->input_value(0));
}

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
    using ov::pass::pattern::consumers_count;
    using ov::pass::pattern::wrap_type;

    // window_reverse: view(B, H / ws, W / ws, ws, ws, -1) with B traced to one.
    const auto split_target = wrap_type<v0::Concat>(is_pinned_target(6));
    const auto split = wrap_type<v1::Reshape>({any_input(), split_target}, consumers_count(1));
    const auto permute =
        wrap_type<v1::Transpose>({split, wrap_type<v0::Constant>(keeps_leading_axis)}, consumers_count(1));

    // window_reverse: view(B, H, W, -1) with B traced to one.
    const auto merge_target = wrap_type<v0::Concat>(is_pinned_target(4));
    const auto merge = wrap_type<v1::Reshape>({permute, merge_target}, consumers_count(1));

    // Reverse cyclic shift of shifted windows.
    const auto shift =
        ov::pass::pattern::optional<v7::Roll>({merge, any_input(), wrap_type<v0::Constant>(excludes_leading_axis)});

    // view(B, H * W, C) taking B from the input shape.
    const auto rebuild_target = wrap_type<v0::Concat>(is_batch_rebuild_target);
    const auto rebuild = wrap_type<v1::Reshape>({shift, rebuild_target});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& matcher) {
        const auto& pattern_map = matcher.get_pattern_value_map();
        const auto batch = pattern_map.at(rebuild_target).get_node()->input_value(0);
        const auto split_target_node = pattern_map.at(split_target).get_node_shared_ptr();
        const auto merge_target_node = pattern_map.at(merge_target).get_node_shared_ptr();
        if (split_target_node->get_element_type() != batch.get_element_type() ||
            merge_target_node->get_element_type() != batch.get_element_type()) {
            return false;
        }

        split_target_node->input(0).replace_source_output(batch);
        merge_target_node->input(0).replace_source_output(batch);
        return true;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(rebuild, matcher_name), callback);
}
