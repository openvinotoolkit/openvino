// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/restore_traced_batch.hpp"

#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
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
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v7 = ov::op::v7;

namespace {

std::optional<int64_t> single_constant_value(const ov::Output<ov::Node>& output) {
    const auto constant = ov::as_type_ptr<v0::Constant>(output.get_node_shared_ptr());
    if (!constant) {
        return std::nullopt;
    }
    const auto values = constant->cast_vector<int64_t>();
    if (values.size() != 1) {
        return std::nullopt;
    }
    return values.front();
}

// The input whose Gather(ShapeOf(input), 0), optionally converted, produces output.
std::shared_ptr<v0::Parameter> batch_source(const ov::Output<ov::Node>& output) {
    auto node = output.get_node_shared_ptr();
    if (ov::is_type<v0::Convert>(node)) {
        node = node->get_input_node_shared_ptr(0);
    }
    const auto gather = ov::as_type_ptr<ov::op::util::GatherBase>(node);
    if (!gather || single_constant_value(gather->input_value(1)) != 0) {
        return nullptr;
    }
    const auto shape_of = gather->get_input_node_shared_ptr(0);
    if (!ov::is_type<v0::ShapeOf>(shape_of) && !ov::is_type<v3::ShapeOf>(shape_of)) {
        return nullptr;
    }
    return ov::as_type_ptr<v0::Parameter>(shape_of->get_input_node_shared_ptr(0));
}

// A target keeping the leading dimension at a constant one while a -1 absorbs the batch instead.
bool is_pinned_leading_target(const std::shared_ptr<ov::Node>& node) {
    const auto target = ov::as_type_ptr<v0::Concat>(node);
    if (!target || target->get_axis() != 0 || target->get_input_size() < 2 ||
        single_constant_value(target->input_value(0)) != 1) {
        return false;
    }
    for (size_t index = 1; index < target->get_input_size(); ++index) {
        if (single_constant_value(target->input_value(index)) == -1) {
            return true;
        }
    }
    return false;
}

bool keeps_leading_axis(const std::shared_ptr<ov::Node>& node) {
    if (ov::is_type<v1::Transpose>(node)) {
        const auto order = ov::as_type_ptr<v0::Constant>(node->get_input_node_shared_ptr(1));
        if (!order) {
            return false;
        }
        const auto values = order->cast_vector<int64_t>();
        return !values.empty() && values.front() == 0;
    }
    if (ov::is_type<v7::Roll>(node)) {
        const auto axes = ov::as_type_ptr<v0::Constant>(node->get_input_node_shared_ptr(2));
        const auto rank = node->get_input_partial_shape(0).rank();
        if (!axes || rank.is_dynamic()) {
            return false;
        }
        for (const auto axis : axes->cast_vector<int64_t>()) {
            if (axis == 0 || axis == -rank.get_length()) {
                return false;
            }
        }
        return true;
    }
    return false;
}

bool depends_on(const std::shared_ptr<ov::Node>& node, const std::shared_ptr<ov::Node>& source) {
    std::unordered_set<std::shared_ptr<ov::Node>> visited;
    std::vector<std::shared_ptr<ov::Node>> pending{node};
    while (!pending.empty()) {
        const auto current = pending.back();
        pending.pop_back();
        if (current == source) {
            return true;
        }
        if (!visited.insert(current).second) {
            continue;
        }
        for (const auto& input : current->input_values()) {
            pending.push_back(input.get_node_shared_ptr());
        }
    }
    return false;
}

}  // namespace

ov::pass::RestoreTracedBatch::RestoreTracedBatch() {
    MATCHER_SCOPE(RestoreTracedBatch);
    const auto rebuild_target = ov::pass::pattern::wrap_type<v0::Concat>();
    const auto rebuild = ov::pass::pattern::wrap_type<v1::Reshape>({ov::pass::pattern::any_input(), rebuild_target});
    const auto restored_targets =
        std::make_shared<std::unordered_map<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>>>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& matcher) {
        const auto target =
            ov::as_type_ptr<v0::Concat>(matcher.get_pattern_value_map().at(rebuild_target).get_node_shared_ptr());
        const auto batch = target->input_value(0);
        const auto parameter = batch_source(batch);
        if (target->get_axis() != 0 || !parameter) {
            return false;
        }

        // Only single-consumer ops keeping the leading axis may separate the pins from the rebuild.
        std::vector<std::shared_ptr<ov::Node>> pins;
        auto node = matcher.get_match_root()->get_input_node_shared_ptr(0);
        while (node->get_output_size() == 1 && node->get_output_target_inputs(0).size() == 1) {
            const bool is_pin = ov::is_type<v1::Reshape>(node) &&
                                is_pinned_leading_target(node->get_input_node_shared_ptr(1)) &&
                                node->get_input_element_type(1) == batch.get_element_type();
            if (!is_pin && !keeps_leading_axis(node)) {
                break;
            }
            if (is_pin) {
                pins.push_back(node);
            }
            node = node->get_input_node_shared_ptr(0);
        }
        if (pins.empty() || !depends_on(pins.back()->get_input_node_shared_ptr(0), parameter)) {
            return false;
        }

        for (const auto& pin : pins) {
            const auto pinned_target = pin->get_input_node_shared_ptr(1);
            const auto [restored_target, inserted] = restored_targets->try_emplace(pinned_target);
            if (inserted) {
                auto inputs = pinned_target->input_values();
                inputs[0] = batch;
                restored_target->second = pinned_target->clone_with_new_inputs(inputs);
                ov::copy_runtime_info(pinned_target, restored_target->second);
            }
            pin->input(1).replace_source_output(restored_target->second);
        }
        return true;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(rebuild, matcher_name), callback);
}
