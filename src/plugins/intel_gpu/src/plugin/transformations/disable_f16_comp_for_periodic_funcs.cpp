// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "openvino/op/util/sub_graph_base.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"
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

#include <unordered_map>
#include <typeindex>

static bool is_non_value_modifying_node(const std::shared_ptr<ov::Node>& node) {
    static const std::unordered_set<std::string> non_value_modifying_nodes = {
        ov::op::v1::Reshape::get_type_info_static().name,
        ov::op::v1::Transpose::get_type_info_static().name,
        ov::op::util::ScatterBase::get_type_info_static().name,
        ov::op::util::ScatterNDBase::get_type_info_static().name,
        ov::op::util::ScatterElementsUpdateBase::get_type_info_static().name,
        ov::op::v8::Slice::get_type_info_static().name,
        ov::op::v1::Broadcast::get_type_info_static().name,
        ov::op::v0::Concat::get_type_info_static().name,
        ov::op::v1::Split::get_type_info_static().name,
        ov::op::v1::StridedSlice::get_type_info_static().name,
        ov::op::v0::Tile::get_type_info_static().name,
        ov::op::v16::Identity::get_type_info_static().name,
        ov::op::v1::Pad::get_type_info_static().name,
        ov::op::util::GatherNDBase::get_type_info_static().name,
        ov::op::util::GatherBase::get_type_info_static().name,
        ov::op::v15::Squeeze::get_type_info_static().name,
        ov::op::v0::Unsqueeze::get_type_info_static().name,
        // Add more node types as needed
    };

    return non_value_modifying_nodes.find(node->get_type_info().name) != non_value_modifying_nodes.end();
}

ov::intel_gpu::DisableFP16CompressionForPeriodicFuncs::DisableFP16CompressionForPeriodicFuncs()
    : ov::pass::MatcherPass() {
    set_name("DisableFP16CompressionForPeriodicFuncs");

    auto cos = ov::pass::pattern::wrap_type<ov::op::v0::Cos>();
    auto sin = ov::pass::pattern::wrap_type<ov::op::v0::Sin>();
    auto sin_cos_func_pattern = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{cos, sin});

    ov::matcher_pass_callback callback = [&](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();

        if (node->get_input_element_type(0) != ov::element::f16
            && node->get_input_element_type(0) != ov::element::f32) {
            return false;
        }

        // Disable FP16 compression for the current node
        ov::disable_fp16_compression(node);

        // Traverse inputs to find the first node that modifies values and disable FP16 compression
        std::function<void(const std::shared_ptr<ov::Node>&)> find_and_disable_fp16_compression = [&](const std::shared_ptr<ov::Node>& current_node) {
            for (const auto& input : current_node->inputs()) {
                auto input_node = input.get_source_output().get_node_shared_ptr();
                if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(input_node)) {
                    // If the next node is a constant, skip it
                    continue;
                }

                if (auto parameter = ov::as_type_ptr<ov::op::v0::Parameter>(input_node)) {
                    // If the next node is a parameter, skip it
                    continue;
                }

                if (ov::fp16_compression_is_disabled(input_node)) {
                    // If FP16 compression is already disabled, continue to the next input
                    return;
                }

                if (is_non_value_modifying_node(input_node)) {
                    // Skip nodes that do not modify values and recursively traverse their inputs
                    return find_and_disable_fp16_compression(input_node);
                } else {
                    // Disable FP16 compression for the first node that modifies values
                    ov::disable_fp16_compression(input_node);
                    return;
                }
            }
        };

        // Start traversal from the current node
        find_and_disable_fp16_compression(node);
        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(sin_cos_func_pattern, "DisableFP16CompressionForPeriodicFuncs");
    register_matcher(m, callback);
}
