// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <stack>

#include "transformations/utils/utils.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/cos.hpp"
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

static bool is_non_value_modifying_node(const std::shared_ptr<ov::Node>& node) {
    return ov::is_type<ov::op::v1::Reshape>(node) ||
           ov::is_type<ov::op::v1::Transpose>(node) ||
           ov::is_type<ov::op::util::ScatterBase>(node) ||
           ov::is_type<ov::op::util::ScatterNDBase>(node) ||
           ov::is_type<ov::op::util::ScatterElementsUpdateBase>(node) ||
           ov::is_type<ov::op::v8::Slice>(node) ||
           ov::is_type<ov::op::v1::Broadcast>(node) ||
           ov::is_type<ov::op::v0::Concat>(node) ||
           ov::is_type<ov::op::v1::Split>(node) ||
           ov::is_type<ov::op::v1::StridedSlice>(node) ||
           ov::is_type<ov::op::v0::Tile>(node) ||
           ov::is_type<ov::op::v16::Identity>(node) ||
           ov::is_type<ov::op::v1::Pad>(node) ||
           ov::is_type<ov::op::util::GatherNDBase>(node) ||
           ov::is_type<ov::op::util::GatherBase>(node) ||
           ov::is_type<ov::op::v15::Squeeze>(node) ||
           ov::is_type<ov::op::v0::Unsqueeze>(node);
}

ov::intel_gpu::DisableFP16CompressionForPeriodicFuncs::DisableFP16CompressionForPeriodicFuncs()
    : ov::pass::MatcherPass() {
    set_name("DisableFP16CompressionForPeriodicFuncs");

    auto cos = ov::pass::pattern::wrap_type<ov::op::v0::Cos>();
    auto sin = ov::pass::pattern::wrap_type<ov::op::v0::Sin>();
    auto sin_cos_func_pattern = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{cos, sin});

    ov::matcher_pass_callback callback = [&](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();

        if (node->get_input_element_type(0) != ov::element::f16) {
            return false;
        }

        // Disable FP16 compression for the current node
        ov::disable_fp16_compression(node);

        // Traverse inputs to find the first node that modifies values and disable FP16 compression
        std::stack<std::shared_ptr<ov::Node>> nodes_to_process;
        nodes_to_process.push(node);

        while (!nodes_to_process.empty()) {
            auto current_node = nodes_to_process.top();
            nodes_to_process.pop();

            OPENVINO_ASSERT((current_node && !current_node->inputs().empty()), "Invalid node or missing inputs during traversal.");

            // Check only the first input
            // Most operations use the first input as the actual data for computation,
            // while other inputs are typically used for shape determination or metadata.
            // Therefore, only the first input is checked to simplify the
            auto input_node = current_node->input(0).get_source_output().get_node_shared_ptr();

            // Skip Constant or Parameter nodes as they do not modify values
            if (ov::as_type_ptr<ov::op::v0::Constant>(input_node) || ov::as_type_ptr<ov::op::v0::Parameter>(input_node)) {
                break;
            }

            // If FP16 compression is already disabled, skip this node
            if (ov::fp16_compression_is_disabled(input_node)) {
                break;
            }

            // If the node does not modify values, add it to the stack for further processing
            if (is_non_value_modifying_node(input_node)) {
                nodes_to_process.push(input_node);
            } else {
                ov::disable_fp16_compression(input_node);

                // Stop further traversal after disabling compression for the first modifying node
                break;
            }
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sin_cos_func_pattern, "DisableFP16CompressionForPeriodicFuncs");
    register_matcher(m, callback);
}
