// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "openvino/op/util/sub_graph_base.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/sinh.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/cosh.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/scatter_base.hpp"
#include "openvino/op/util/scatter_nd_base.hpp"
#include "openvino/op/util/scatter_elements_update_base.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/identity.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/op/util/gather_nd_base.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "disable_f16_comp_for_periodic_funcs.hpp"

#include <unordered_map>
#include <typeindex>

static bool is_non_value_modifying_node(const std::shared_ptr<ov::Node>& node) {
    static const std::unordered_map<std::type_index, bool> node_behavior_registry = {
        {typeid(ov::op::v1::Reshape), true},
        {typeid(ov::op::v1::Transpose), true},
        {typeid(ov::op::util::ScatterBase), true},
        {typeid(ov::op::util::ScatterNDBase), true},
        {typeid(ov::op::util::ScatterElementsUpdateBase), true},
        {typeid(ov::op::v8::Slice), true},
        {typeid(ov::op::v1::Broadcast), true},
        {typeid(ov::op::v0::Concat), true},
        {typeid(ov::op::v1::Split), true},
        {typeid(ov::op::v1::StridedSlice), true},
        {typeid(ov::op::v0::Tile), true},
        {typeid(ov::op::v16::Identity), true},
        {typeid(ov::op::v1::Pad), true},
        {typeid(ov::op::util::GatherNDBase), true},
        {typeid(ov::op::util::GatherBase), true},
        {typeid(ov::op::v15::Squeeze), true},
        {typeid(ov::op::v0::Unsqueeze), true},
        // Add more node types as needed
    };

    auto it = node_behavior_registry.find(typeid(*node));
    return it != node_behavior_registry.end() && it->second;
}

ov::intel_gpu::DisableFP16CompressionForPeriodicFuncs::DisableFP16CompressionForPeriodicFuncs()
    : ov::pass::MatcherPass() {
    set_name("DisableFP16CompressionForPeriodicFuncs");

    auto cos = ov::pass::pattern::wrap_type<ov::op::v0::Cos>();
    auto sin = ov::pass::pattern::wrap_type<ov::op::v0::Sin>();
    auto cosh = ov::pass::pattern::wrap_type<ov::op::v0::Cosh>();
    auto sinh = ov::pass::pattern::wrap_type<ov::op::v0::Sinh>();
    auto periodic_func_pattern = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{cos, sin, cosh, sinh});

    ov::matcher_pass_callback callback = [&](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();

        if (node->get_input_element_type(0) != ov::element::f16
            && node->get_input_element_type(0) != ov::element::f32) {
            return false;
        }

        // Disable FP16 compression for the current node
        ov::disable_fp16_compression(node);

        // Traverse inputs to find the first node that modifies values
        std::function<bool(const std::shared_ptr<ov::Node>&)> traverse_inputs = [&](const std::shared_ptr<ov::Node>& current_node) -> bool {
            for (const auto& input : current_node->inputs()) {
                auto next_node = input.get_source_output().get_node_shared_ptr();
                if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(next_node)) {
                    // If the next node is a constant, skip it
                    continue;
                }

                if (auto parameter = ov::as_type_ptr<ov::op::v0::Parameter>(next_node)) {
                    // If the next node is a parameter, skip it
                    continue;
                }

                if (ov::fp16_compression_is_disabled(next_node)) {
                    // If FP16 compression is already disabled, continue to the next input
                    return true;
                }

                if (is_non_value_modifying_node(next_node)) {
                    // Skip nodes that do not modify values and recursively traverse their inputs
                    return traverse_inputs(next_node);
                } else {
                    // Disable FP16 compression for the first node that modifies values
                    ov::disable_fp16_compression(next_node);
                    return true;
                }
            }
            return false; // No node found that modifies values
        };
        // Start traversal from the current node
        return traverse_inputs(node);
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(periodic_func_pattern, "DisableFP16CompressionForPeriodicFuncs");
    register_matcher(m, callback);
}
