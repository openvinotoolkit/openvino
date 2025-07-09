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
#include "openvino/op/cos.hpp"
#include "disable_f16_comp_for_periodic_funcs.hpp"

ov::intel_gpu::DisableFP16CompressionForPeriodicFuncs::DisableFP16CompressionForPeriodicFuncs()
    : ov::pass::MatcherPass() {
    set_name("DisableFP16CompressionForPeriodicFuncs");

    auto cos = ov::pass::pattern::wrap_type<ov::op::v0::Cos>();
    auto sin = ov::pass::pattern::wrap_type<ov::op::v0::Sin>();
    auto cos_sin_pattern = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{cos, sin});

    ov::matcher_pass_callback callback = [&](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();

        if (node->get_input_element_type(0) != ov::element::f16
            && node->get_input_element_type(0) != ov::element::f32) {
            return false;
        }

        // Disable FP16 compression for the current node
        ov::disable_fp16_compression(node);

        // List of node types to skip during input traversal
        std::unordered_set<std::string> skip_node_types = {
            "Const",          // Constant values
            "Reshape",        // Reshapes tensor dimensions
            "Transpose",      // Transposes tensor axes
            "ScatterNDUpdate",// Updates tensor values at specific indices
            "Slice",          // Extracts a slice of the tensor
            "Broadcast",      // Broadcasts tensor to a new shape
            "Concat",         // Concatenates tensors along a specific axis
            "Split",          // Splits tensor into multiple parts
            "StridedSlice",   // Extracts slices with strides
            "Tile",           // Repeats tensor along specified dimensions
            "Identity",       // Passes input directly as output
            "Pad",            // Adds padding to tensor
            "Gather",         // Gathers elements along an axis
            "GatherND",       // Gathers elements using multi-dimensional indices
            "Rank",           // Returns the rank of the tensor
            "Squeeze",        // Removes dimensions of size 1
            "Unsqueeze",      // Adds dimensions of size 1
            "ReduceSum",      // Reduces tensor by summing along axes (doesn't modify individual values)
            "ReduceMean",     // Reduces tensor by averaging along axes
            "ReduceMax",      // Reduces tensor by taking the maximum along axes
            "ReduceMin",      // Reduces tensor by taking the minimum along axes
            "ReduceProd"      // Reduces tensor by taking the product along axes
        };

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
                if (skip_node_types.find(next_node->get_type_info().name) != skip_node_types.end()) {
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
    auto m = std::make_shared<ov::pass::pattern::Matcher>(cos_sin_pattern, "DisableFP16CompressionForPeriodicFuncs");
    register_matcher(m, callback);
}
