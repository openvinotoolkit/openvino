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

// Overload << operator for vectors
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

std::vector<int32_t> compute_level_start_index(const std::vector<int32_t>& spatial_shapes) {
    std::vector<int32_t> level_start_index;
    size_t num_level = spatial_shapes.size() >> 1;

    if(spatial_shapes.empty()) return {};
    
    // Always start with 0 for the first level
    level_start_index.reserve(num_level + 1);
    level_start_index.push_back(0);
    
    int32_t cumulative = 0;
    // Process all levels except the last one
    for (size_t i = 0; i < num_level - 1; ++i) {
        cumulative += spatial_shapes[2 * i] * spatial_shapes[2 * i + 1];
        level_start_index.push_back(cumulative);
    }
    
    std::cout << "spatial_shapes = " << spatial_shapes << ", level_start_index =  " << level_start_index << std::endl;
    return level_start_index;
}

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
    auto attn_Reshape_5 = wrap_type<Reshape>({attn_Transpose, any_input(/*pattern::shape_matches("[..., img_h, img_w]" && pattern::rank_equals(4)*/)});
    
    auto attn_Gather_9 = wrap_type<Gather>({input_attn_offsets, any_input(), any_input()});
    // auto attn_squeeze_optional = optional<Squeeze>({attn_Gather_9, any_input()});
    auto attn_squeeze_optional = wrap_type<Squeeze, Reshape>({attn_Gather_9, any_input()});
    auto attn_Transpose_1 = wrap_type<Transpose>({attn_squeeze_optional, any_input()});
    auto attn_Reshape_6 = wrap_type<Reshape>({attn_Transpose_1, any_input()});

    auto attn_GridSample = wrap_type<GridSample>({attn_Reshape_5, attn_Reshape_6});
    auto attn_Unsqueeze_31 = wrap_type<Reshape>({attn_GridSample, any_input()});

    auto block = std::make_shared<pattern::op::Block>(OutputVector{input_attn_value, input_attn_offsets}, OutputVector{attn_Unsqueeze_31}, "grid_sample_block");

    REGISTER_ANCHORS(block, attn_Unsqueeze_31, attn_Reshape_5);

    return block;
}

}  // namespace

MultiScaleDeformableAttnFusion::MultiScaleDeformableAttnFusion() : MultiMatcher("MultiScaleDeformableAttnFusion") {
    using namespace ov::opset12;

    // Pattern 1
    auto attn_value_input = any_input();
    auto attn_offsets_input = any_input();
    auto grid_sampler_block = grid_sample_block(attn_value_input, attn_offsets_input);

    // std::cout << "wzx debug hit in in" << __LINE__ << std::endl;

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

    auto callback = [OV_CAPTURE_CPY_AND_THIS](const std::unordered_map<std::shared_ptr<Node>, std::vector<PatternValueMap>>& matches) {
        // std::cout << "wzx debug return directly" << std::endl;
        // return;
        // std::cout << "wzx debug hit in in" << __LINE__ << ", matches.size()=" << matches.size() << std::endl;
        if (matches.size() != 2) {
            return;
        }

        // std::cout << "wzx debug hit in in" << __LINE__ << std::endl;

        std::unordered_map<Node*, const PatternValueMap*> node_to_output_proj_pm;
        for (const auto& pm : matches.at(attn_output_proj_MatMul_transpose_a)) {
            auto root = pm.at(attn_output_proj_MatMul_transpose_a).get_node();
            node_to_output_proj_pm[root] = &pm;
            // std::cout << "wzx debug hit in in" << __LINE__ << ", root=" << root->get_friendly_name() << std::endl;
        }

        std::unordered_set<Node*> post_sdpa_proj;
        std::unordered_map<Node*, const PatternValueMap*> node_to_grid_sampler_pm;
        for (const auto& pm : matches.at(grid_sampler_block)) {
            auto root = pm.at(grid_sampler_block).get_node_shared_ptr();
            auto block =
                std::dynamic_pointer_cast<ov::pass::pattern::op::Block>(root);
            auto anchor = block->get_anchor("attn_Unsqueeze_31", pm).value().get_node_shared_ptr();
            node_to_grid_sampler_pm[anchor.get()] = &pm;
            // std::cout << "wzx debug hit in in" << __LINE__ << ", root=" << root->get_friendly_name() <<
                    // ", anchor=" << anchor->get_friendly_name() << std::endl;
        }

        // std::cout << "wzx debug hit in in" << __LINE__ << std::endl;
        auto retreive_spatial_shapes = [=](const std::shared_ptr<Node> attn_Concat_17_node) -> std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>> {
            // size_t num_level = 4;
            // auto spatial_shapes_node = Constant::create(element::i32, Shape{num_level, 2}, { 100, 167,
            //                                                                                         50,  84,
            //                                                                                         25,  42,                                                                                
            //                                                                                         13,  21});
            // auto level_start_index_node = Constant::create(element::i32, Shape{num_level,}, {0, 16700, 20900, 21950});
            // return {spatial_shapes_node, level_start_index_node};

            size_t num_level = attn_Concat_17_node->get_input_size();
            std::vector<int32_t> spatial_shapes;
            spatial_shapes.resize(num_level * 2);
            std::cout << "================= num_level = " << num_level << std::endl;
            for (size_t i = 0; i < num_level; i++) {
                std::cout << "================= current_level = " << i << std::endl;
                auto input_node = attn_Concat_17_node->input_value(i).get_node();

                OPENVINO_ASSERT(node_to_grid_sampler_pm.count(input_node) > 0);

                const auto* grid_sampler_pm = node_to_grid_sampler_pm.at(input_node);

                auto grid_sampler_block_node = ov::as_type_ptr<pattern::op::Block>(grid_sampler_pm->at(grid_sampler_block).get_node_shared_ptr());
                OPENVINO_ASSERT(grid_sampler_block_node != nullptr);

                auto reshape_anchor = grid_sampler_block_node->get_anchor("attn_Reshape_5", *grid_sampler_pm).value().get_node_shared_ptr();
                const auto target_shape_contant = ov::as_type_ptr<v0::Constant>(reshape_anchor->get_input_node_shared_ptr(1));
                OPENVINO_ASSERT(target_shape_contant != nullptr);

                auto spatial_shape_value = target_shape_contant->cast_vector<int64_t>();
                OPENVINO_ASSERT(spatial_shape_value.size() == 4);
                spatial_shapes[i * 2] = static_cast<int32_t>(spatial_shape_value[2]);
                spatial_shapes[i * 2 + 1] = static_cast<int32_t>(spatial_shape_value[3]);
            }

            OPENVINO_ASSERT(!spatial_shapes.empty());
            auto level_start_index = compute_level_start_index(spatial_shapes);

            auto spatial_shapes_node = Constant::create(element::i32, Shape{num_level, 2}, spatial_shapes.data());
            auto level_start_index_node = Constant::create(element::i32, Shape{num_level,}, level_start_index.data());
            return {spatial_shapes_node, level_start_index_node};
        };

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
                // # (num_level, 2)
                // spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
                // level_start_index = torch.cat((
                //     spatial_shapes.new_zeros((1, )),  # (num_level)
                //     spatial_shapes.prod(1).cumsum(0)[:-1]))
                std::cout << "wzx debug hit in in" << __LINE__ << ", " << grid_sampler_pm << std::endl;
                auto [spatial_shapes_input_node, level_start_index_input_node] = retreive_spatial_shapes(attn_Concat_17_node);

                //
                std::cout << "wzx debug hit in in" << __LINE__ << ", " << grid_sampler_pm << std::endl;
                auto msda_node = std::make_shared<ov::op::internal::MSDA>(OutputVector{attn_value_input_node,
                                                                                    spatial_shapes_input_node,
                                                                                    level_start_index_input_node,
                                                                                    attn_offsets_input_node,
                                                                                    attn_weight_input_node});
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