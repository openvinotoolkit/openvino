// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/restore_traced_batch.hpp"

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/roll.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v7 = ov::op::v7;
namespace op_util = ov::op::util;

namespace {

// A target keeping the leading dimension at a constant one while a -1 absorbs the batch instead.
bool is_pinned_leading_target(const std::shared_ptr<ov::Node>& node) {
    const auto target = ov::as_type_ptr<v0::Concat>(node);
    if (!target || target->get_axis() != 0 || target->get_input_size() < 2 ||
        !op_util::has_constant_value<int64_t>(target->get_input_node_shared_ptr(0), 1)) {
        return false;
    }
    const auto inputs = target->input_values();
    return std::any_of(inputs.begin() + 1, inputs.end(), [](const ov::Output<ov::Node>& input) {
        return op_util::has_constant_value<int64_t>(input.get_node_shared_ptr(), -1);
    });
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
        auto values = axes->cast_vector<int64_t>();
        ov::util::normalize_axes(values, rank.get_length());
        return std::find(values.begin(), values.end(), 0) == values.end();
    }
    return false;
}

}  // namespace

ov::pass::RestoreTracedBatch::RestoreTracedBatch() : MultiMatcher("RestoreTracedBatch") {
    MATCHER_SCOPE(RestoreTracedBatch);
    // Batch of an input: Gather(ShapeOf(Parameter), 0), optionally converted to the Reshape target type.
    const auto batch_input = ov::pass::pattern::wrap_type<v0::Parameter>();
    const auto batch_gather = ov::pass::pattern::wrap_type<op_util::GatherBase>(
        {ov::pass::pattern::wrap_type<v0::ShapeOf, v3::ShapeOf>({batch_input}),
         ov::pass::pattern::wrap_type<v0::Constant>(ov::pass::pattern::value_matches("0")),
         ov::pass::pattern::any_input()});
    const auto batch = ov::pass::pattern::wrap_type<v0::Convert>({batch_gather}) | batch_gather;

    // Rebuild: a Reshape whose target is a Concat, checked in the callback to start with a batch.
    const auto rebuild_target = ov::pass::pattern::wrap_type<v0::Concat>();
    const auto rebuild = ov::pass::pattern::wrap_type<v1::Reshape>({ov::pass::pattern::any_input(), rebuild_target});

    auto callback = [=](const std::unordered_map<std::shared_ptr<ov::Node>,
                                                 std::vector<ov::pass::pattern::PatternValueMap>>& matches) {
        const auto batch_matches = matches.find(batch);
        const auto rebuild_matches = matches.find(rebuild);
        if (batch_matches == matches.end() || rebuild_matches == matches.end()) {
            return;
        }
        std::unordered_map<ov::Node*, ov::Node*> input_of_batch;
        for (const auto& match : batch_matches->second) {
            input_of_batch.emplace(match.at(batch).get_node(), match.at(batch_input).get_node());
        }

        std::unordered_map<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> restored_targets;
        for (const auto& match : rebuild_matches->second) {
            const auto target = ov::as_type_ptr<v0::Concat>(match.at(rebuild_target).get_node_shared_ptr());
            const auto leading_dimension = target->input_value(0);
            const auto parameter = input_of_batch.find(leading_dimension.get_node());
            if (target->get_axis() != 0 || parameter == input_of_batch.end()) {
                continue;
            }

            // Only single-consumer ops keeping the leading axis may separate the pins from the rebuild.
            std::vector<std::shared_ptr<ov::Node>> pins;
            auto node = match.at(rebuild).get_node()->get_input_node_shared_ptr(0);
            while (node->get_output_size() == 1 && node->get_output_target_inputs(0).size() == 1) {
                const bool is_pin = ov::is_type<v1::Reshape>(node) &&
                                    is_pinned_leading_target(node->get_input_node_shared_ptr(1)) &&
                                    node->get_input_element_type(1) == leading_dimension.get_element_type();
                if (!is_pin && !keeps_leading_axis(node)) {
                    break;
                }
                if (is_pin) {
                    pins.push_back(node);
                }
                node = node->get_input_node_shared_ptr(0);
            }
            if (pins.empty()) {
                continue;
            }
            std::unordered_set<ov::Node*> pinned_data_sources;
            op_util::visit_path(
                pins.back()->get_input_node_ptr(0),
                pinned_data_sources,
                [](ov::Node*) {},
                [](ov::Node*) {
                    return false;
                });
            if (!pinned_data_sources.count(parameter->second)) {
                continue;
            }

            for (const auto& pin : pins) {
                const auto pinned_target = pin->get_input_node_shared_ptr(1);
                auto& restored_target = restored_targets[pinned_target];
                if (!restored_target) {
                    auto inputs = pinned_target->input_values();
                    inputs[0] = leading_dimension;
                    restored_target = pinned_target->clone_with_new_inputs(inputs);
                    ov::copy_runtime_info(pinned_target, restored_target);
                }
                pin->input(1).replace_source_output(restored_target);
            }
        }
    };

    register_patterns({batch, rebuild}, callback);
}
