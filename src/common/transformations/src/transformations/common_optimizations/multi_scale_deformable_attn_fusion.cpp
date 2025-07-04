// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/multi_scale_deformable_attn_fusion.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/block_collection.hpp"
#include "transformations/utils/utils.hpp"

#include "transformations/utils/block_collection.hpp"

#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/power.hpp"
#include "openvino/pass/pattern/op/block.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "openvino/opsets/opset12.hpp"

#include "ov_ops/msda.hpp"

using namespace ov;
using namespace ov::op;
using namespace ov::pass;
using namespace ov::pass::pattern;
using namespace ov::opset12;

namespace {

ov::pass::pattern::op::Predicate check_input(std::shared_ptr<Node> expected_input) {
    return ov::pass::pattern::op::Predicate(
        [=](const Output<Node>& output) -> bool {
            auto graph_node = output.get_node_shared_ptr();
            auto pattern_node = expected_input.get();
            for (size_t i = 0; i < graph_node->get_input_size(); i++) {
                auto input_node = graph_node->input_value(i).get_node();
                if (pattern_node == input_node) return true;
            }
            
            return false;
        },
        "check_input");
}

std::shared_ptr<ov::Node> grid_sample_block(const std::shared_ptr<ov::Node>& input_attn_value, const std::shared_ptr<ov::Node>& input_attn_offsets) {
    auto attn_Slice = wrap_type<StridedSlice>({input_attn_value, any_input(), any_input(), any_input()});
    auto attn_Reshape_4 = wrap_type<Reshape>({attn_Slice, any_input()});
    auto attn_Transpose = wrap_type<Transpose>({attn_Reshape_4, any_input()});
    auto attn_Reshape_5 = wrap_type<Reshape>({attn_Transpose, any_input()});
    
    auto attn_Gather_9 = wrap_type<Gather>({input_attn_offsets, any_input(), any_input()});
    auto attn_squeeze_optional = optional<Squeeze>({attn_Gather_9, any_input()});
    auto attn_Transpose_1 = wrap_type<Transpose>({attn_squeeze_optional, any_input()});
    auto attn_Reshape_6 = wrap_type<Reshape>({attn_Transpose_1, any_input()});

    auto attn_GridSample = wrap_type<GridSample>({attn_Reshape_5, attn_Reshape_6});
    auto attn_Unsqueeze_31 = wrap_type<Reshape>({attn_GridSample, any_input()});

    auto block = std::make_shared<pattern::op::Block>(OutputVector{input_attn_value, input_attn_offsets}, OutputVector{attn_Unsqueeze_31}, "grid_sample_block");

    REGISTER_ANCHORS(block, attn_Unsqueeze_31);

    return block;
}

}  // namespace

MultiScaleDeformableAttnFusion::MultiScaleDeformableAttnFusion() : MultiMatcher("MultiScaleDeformableAttnFusion") {
    using namespace ov::opset12;

    // Pattern 1
    auto attn_value_input = any_input();
    auto attn_offsets_input = any_input();
    auto grid_sampler_block = grid_sample_block(attn_value_input, attn_offsets_input);

    std::cout << "wzx debug hit in in" << __LINE__ << std::endl;

    // Pattern 2
    auto attn_weight_input = any_input();
    auto attn_Transpose_8 = wrap_type<Transpose>({attn_weight_input, any_input()});
    auto attn_Reshape_16 = wrap_type<Reshape>({attn_Transpose_8, any_input()});

    //({flatten_Slice_1194, {-1}}, {{"axis", 0}});
    // ({Unsqueeze_65524 | Unsqueeze_28998, Unsqueeze_65525 | Unsqueeze_28999},
    // wrap_type<opset1::Concat>(pattern::consumers_count(1));
    auto attn_Concat_17 = wrap_type<Concat>({{"axis", -2}});
    auto attn_Reshape_17 = wrap_type<Reshape>({attn_Concat_17, any_input()});

    auto attn_Mul_3 = wrap_type<Multiply>({attn_Reshape_17, attn_Reshape_16});
    auto attn_ReduceSum = wrap_type<ReduceSum>({attn_Mul_3, any_input()});
    auto attn_Reshape_18 = wrap_type<Reshape>({attn_ReduceSum, any_input()});
    auto attn_output_proj_MatMul_transpose_a = wrap_type<Transpose>({attn_Reshape_18, any_input()});

    auto callback = [=](const std::unordered_map<std::shared_ptr<Node>, std::vector<PatternValueMap>>& matches) {
        std::cout << "wzx debug hit in in" << __LINE__ << ", matches.size()=" << matches.size() << std::endl;
        if (matches.size() != 2) {
            return;
        }

        std::cout << "wzx debug hit in in" << __LINE__ << std::endl;

        std::unordered_map<Node*, const PatternValueMap*> node_to_output_proj_pm;
        for (const auto& pm : matches.at(attn_output_proj_MatMul_transpose_a)) {
            auto root = pm.at(attn_output_proj_MatMul_transpose_a).get_node();
            node_to_output_proj_pm[root] = &pm;
            std::cout << "wzx debug hit in in" << __LINE__ << ", root=" << root->get_friendly_name() << std::endl;
        }

        std::unordered_map<Node*, const PatternValueMap*> node_to_grid_sampler_pm;
        for (const auto& pm : matches.at(grid_sampler_block)) {
            auto root = pm.at(grid_sampler_block).get_node_shared_ptr();
            auto block =
                std::dynamic_pointer_cast<ov::pass::pattern::op::Block>(root);
            auto anchor = block->get_anchor("attn_Unsqueeze_31", pm).value().get_node_shared_ptr();
            node_to_grid_sampler_pm[anchor.get()] = &pm;
            std::cout << "wzx debug hit in in" << __LINE__ << ", root=" << root->get_friendly_name() <<
                    ", anchor=" << anchor->get_friendly_name() << std::endl;
        }

        std::cout << "wzx debug hit in in" << __LINE__ << std::endl;

        for (const auto& [output_proj_root, output_proj_pm] : node_to_output_proj_pm) {
            OPENVINO_ASSERT(output_proj_pm->count(attn_Concat_17) > 0);

            auto attn_Concat_17_node = output_proj_pm->at(attn_Concat_17).get_node_shared_ptr();
            auto input_node = attn_Concat_17_node->input_value(0).get_node();

            std::cout << "wzx debug hit in in" << __LINE__ << ", " << output_proj_root->get_friendly_name() << std::endl;

            if (node_to_grid_sampler_pm.count(input_node)) {
                const auto* grid_sampler_pm = node_to_grid_sampler_pm.at(input_node);

                std::cout << "wzx debug hit in in" << __LINE__ << ", " << grid_sampler_pm << std::endl;
                //
                // auto attn_value_input_node = grid_sampler_pm->at(attn_value_input);
                // auto attn_offsets_input_node = grid_sampler_pm->at(attn_offsets_input);
                auto attn_weight_input_node = output_proj_pm->at(attn_weight_input);

                OPENVINO_ASSERT(grid_sampler_pm->count(grid_sampler_block) > 0);
                auto grid_sampler_block_node = ov::as_type_ptr<pattern::op::Block>(grid_sampler_pm->at(grid_sampler_block).get_node_shared_ptr());
                OPENVINO_ASSERT(grid_sampler_block_node != nullptr);
                auto attn_value_input_node = grid_sampler_block_node->get_inputs()[0].get_node_shared_ptr();
                auto attn_offsets_input_node = grid_sampler_block_node->get_inputs()[1].get_node_shared_ptr();
                //
                auto msda_node = std::make_shared<ov::op::internal::MSDA>(OutputVector{attn_value_input_node, attn_offsets_input_node, attn_weight_input_node});
                auto consumers = output_proj_root->get_output_target_inputs(0);
                for (auto consumer: consumers) {
                    consumer.replace_source_output(msda_node);
                }

                std::cout << "wzx debug hit in in" << __LINE__ << ", " << input_node->get_friendly_name() << std::endl;
            }            
        }
    };

    register_patterns({grid_sampler_block, attn_output_proj_MatMul_transpose_a}, callback, true);
}