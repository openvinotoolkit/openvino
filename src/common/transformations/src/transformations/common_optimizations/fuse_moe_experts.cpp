// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_moe_experts.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <tuple>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rank.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/one_hot.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/print_model.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/pass/pattern/op/block.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace pass {

using namespace ov::gen_pattern;
using namespace ov::pass;
using namespace ov::pass::pattern;

namespace {

struct expert_data {
    std::shared_ptr<ov::pass::pattern::op::Block> expert_block;
    std::shared_ptr<Node> gate_proj_weight;
    std::shared_ptr<Node> up_proj_weight;
    std::shared_ptr<Node> down_proj_weight;
    size_t expert_id;
};

std::shared_ptr<pattern::op::Block> mlp3_no_bias_swiglu_block(const Output<Node>& permute_Transpose, // Transpose -> OneHot -> TopK -> Softmax -> MatMul -> Hidden States
                                                const Output<Node>& unsqueeze_Unsqueeze, // Unsqueeze -> Reshape -> Hidden States
                                                const Output<Node>& index_Split_out_1, // Split -> Unsqueeze -> Divide -> TopK -> Softmax -> MatMul -> Hidden States
                                                const Output<Node>& index_Reshape // Reshape -> Divide -> TopK -> Softmax -> MatMul -> Hidden States
                                               ) {

    auto index_add__ScatterElementsUpdate_2 = pattern::any_input();
    auto shape_const = wrap_type<ov::op::v0::Constant>();
    auto expert_id = wrap_type<ov::op::v0::Constant>();
    // Gather input 1 is expert IDx
    auto select_Gather_1 = wrap_type<ov::op::v8::Gather>({permute_Transpose, expert_id, 0}, {{"batch_dims", 0}});
    auto squeeze_Squeeze_1 = wrap_type<ov::op::v0::Squeeze>({select_Gather_1, 0});
    auto ListUnpack_NonZero_1 = wrap_type<ov::op::v3::NonZero>({squeeze_Squeeze_1}, {{"output_type", "i64"}});
    auto ListUnpack_Split_1 = wrap_type<ov::op::v1::Split>({ListUnpack_NonZero_1, 0}, {{"num_splits", 2}});
    ListUnpack_Split_1->set_output_size(2);
    auto ListUnpack_Squeeze_0_1 = wrap_type<ov::op::v0::Squeeze>({ListUnpack_Split_1->output(1), 0});
    auto index_add__Convert_1 = wrap_type<ov::op::v0::Convert>({ListUnpack_Squeeze_0_1}, {{"destination_type", "i32"}});
    auto index_add__Reshape_1 = wrap_type<ov::op::v1::Reshape>({index_add__Convert_1, {-1, 1}}, {{"special_zero", false}});
    // Input 0 - any scatter or broadcast?
    auto index_add__Slice_1 =
        wrap_type<ov::op::v8::Slice>({index_add__ScatterElementsUpdate_2, {0, 0}, {1, INT_MAX}, {1, 1}, {0, 1}});
    auto index_add__ShapeOf_14 = wrap_type<ov::op::v3::ShapeOf>({index_add__Slice_1}, {{"output_type", "i32"}});
    auto index_add__Broadcast_16 =
        wrap_type<ov::op::v3::Broadcast>({index_add__Reshape_1, index_add__ShapeOf_14}, {{"mode", "bidirectional"}});
    auto index_Gather_2 =
        wrap_type<ov::op::v8::Gather>({unsqueeze_Unsqueeze, index_add__Convert_1, 1}, {{"batch_dims", 0}});
    auto reshape_Reshape_1 = wrap_type<ov::op::v1::Reshape>({index_Gather_2, shape_const}, {{"special_zero", true}});
    auto gate_proj_weight = makePattern(ov::Rank(2));
    auto linear_MatMul_gate =
        wrap_type<ov::op::v0::MatMul>({reshape_Reshape_1, gate_proj_weight},
                                  {{"transpose_a", false}, {"transpose_b", true}});
    auto silu_Swish = wrap_type<ov::op::v4::Swish>({linear_MatMul_gate});
    auto up_proj_weight = makePattern(ov::Rank(2));
    auto linear_MatMul_up =
        wrap_type<ov::op::v0::MatMul>({reshape_Reshape_1, up_proj_weight},
                                  {{"transpose_a", false}, {"transpose_b", true}});
    auto mul_Multiply = wrap_type<ov::op::v1::Multiply>({silu_Swish, linear_MatMul_up}, {{"auto_broadcast", "numpy"}});
    auto down_proj_weight = makePattern(ov::Rank(2));
    auto linear_MatMul_down = wrap_type<ov::op::v0::MatMul>({mul_Multiply, down_proj_weight},
                                                   {{"transpose_a", false}, {"transpose_b", true}});
    auto ListUnpack_Squeeze_1 = wrap_type<ov::op::v0::Squeeze>({ListUnpack_Split_1->output(0), 0});
    auto index_Convert_4 = wrap_type<ov::op::v0::Convert>({ListUnpack_Squeeze_1}, {{"destination_type", "i32"}});
    auto index_Multiply_1 =
        wrap_type<ov::op::v1::Multiply>({index_add__Convert_1, index_Split_out_1}, {{"auto_broadcast", "numpy"}});
    auto index_Add_1 = wrap_type<ov::op::v1::Add>({index_Convert_4, index_Multiply_1}, {{"auto_broadcast", "numpy"}});
    auto index_Gather_3 = wrap_type<ov::op::v8::Gather>({index_Reshape, index_Add_1, 0}, {{"batch_dims", 0}});
    auto index_Reshape_8_1 = wrap_type<ov::op::v1::Reshape>({index_Gather_3, {0, 1}}, {{"special_zero", true}});
    auto mul_Multiply_2 =
        wrap_type<ov::op::v1::Multiply>({linear_MatMul_down, index_Reshape_8_1}, {{"auto_broadcast", "numpy"}});
    auto index_add__Broadcast_17 =
        wrap_type<ov::op::v3::Broadcast>({mul_Multiply_2, index_add__ShapeOf_14}, {{"mode", "bidirectional"}});
    // Input 0 - any scatter or broadcast?
    auto index_add__ScatterElementsUpdate_5 = wrap_type<ov::op::v12::ScatterElementsUpdate>(
        {index_add__ScatterElementsUpdate_2, index_add__Broadcast_16, index_add__Broadcast_17, 0},
        {{"reduction", "sum"}, {"use_init_val", true}});
    auto block = std::make_shared<pattern::op::Block>(OutputVector{permute_Transpose, unsqueeze_Unsqueeze, index_Split_out_1, index_Reshape}, OutputVector{index_add__ScatterElementsUpdate_5}, "expert_block");
    REGISTER_ANCHORS(block, expert_id, gate_proj_weight, up_proj_weight, down_proj_weight);
    return block;
}

}  // namespace

ov::pass::FuseMOEExperts::FuseMOEExperts() : MultiMatcher("FuseMOEExperts") {
    auto view_Reshape = pattern::any_input();
    auto self_model_layers_1_mlp_gate_weight = pattern::any_input();
    auto shape_const = wrap_type<ov::op::v0::Constant>();
    auto expert_num = wrap_type<ov::op::v0::Constant>();
    auto num_topk = wrap_type<ov::op::v0::Constant>();
    // auto linear_MatMul = wrap_type<ov::op::v0::MatMul>({view_Reshape, self_model_layers_1_mlp_gate_weight}, {{"transpose_a", false}, {"transpose_b", true}});
    auto linear_MatMul = pattern::any_input();
    auto softmax_Softmax = wrap_type<ov::op::v8::Softmax>({linear_MatMul}, {{"axis", 1}});
    auto topk_TopK = wrap_type<ov::op::v11::TopK>({softmax_Softmax, num_topk}, {{"axis", -1}, {"mode", "max"}, {"sort", "value"}, {"index_element_type", "i64"}, {"stable", false}});
    topk_TopK->set_output_size(2);
    auto one_hot_OneHot = wrap_type<ov::op::v1::OneHot>({topk_TopK->output(1), expert_num, 1, 0}, {{"axis", 2}});
    auto permute_Transpose = wrap_type<ov::op::v1::Transpose>({one_hot_OneHot, {2,1,0}});
    auto select_Gather = wrap_type<ov::op::v8::Gather>({permute_Transpose, 0, 0}, {{"batch_dims", 0}});
    auto squeeze_Squeeze = wrap_type<ov::op::v0::Squeeze>({select_Gather, 0});
    auto ListUnpack_NonZero = wrap_type<ov::op::v3::NonZero>({squeeze_Squeeze}, {{"output_type", "i64"}});
    auto ListUnpack_Split = wrap_type<ov::op::v1::Split>({ListUnpack_NonZero, 0}, {{"num_splits", 2}});
    ListUnpack_Split->set_output_size(2);
    auto ListUnpack_Squeeze_0 = wrap_type<ov::op::v0::Squeeze>({ListUnpack_Split->output(1), 0});
    auto index_add__Convert = wrap_type<ov::op::v0::Convert>({ListUnpack_Squeeze_0}, {{"destination_type", "i32"}});
    auto index_add__Reshape = wrap_type<ov::op::v1::Reshape>({index_add__Convert, {-1,1}}, {{"special_zero", false}});
    auto index_add__Slice = wrap_type<ov::op::v8::Slice>({pattern::any_input(), {0,0}, {1,INT_MAX}, {1,1}, {0,1}});
    auto index_add__ShapeOf_6 = wrap_type<ov::op::v3::ShapeOf>({index_add__Slice}, {{"output_type", "i32"}});
    auto index_add__Broadcast_7 = wrap_type<ov::op::v3::Broadcast>({index_add__Reshape, index_add__ShapeOf_6}, {{"mode", "bidirectional"}});
    auto unsqueeze_Unsqueeze = wrap_type<ov::op::v0::Unsqueeze>({view_Reshape, 0});
    auto index_Gather = wrap_type<ov::op::v8::Gather>({unsqueeze_Unsqueeze, index_add__Convert, 1}, {{"batch_dims", 0}});
    auto reshape_Reshape = wrap_type<ov::op::v1::Reshape>({index_Gather, shape_const}, {{"special_zero", true}});
    auto sum_ReduceSum = wrap_type<ov::op::v1::ReduceSum>({topk_TopK->output(0), {-1}}, {{"keep_dims", true}});
    auto div__Divide = wrap_type<ov::op::v1::Divide>({topk_TopK->output(0), sum_ReduceSum}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
    auto unsqueeze_Unsqueeze_1 = makeOP<opset1::Unsqueeze>({div__Divide, 2});
    auto index_ShapeOf_1 = wrap_type<ov::op::v3::ShapeOf>({unsqueeze_Unsqueeze_1}, {{"output_type", "i32"}});
    auto index_Split = wrap_type<ov::op::v1::Split>({index_ShapeOf_1, 0}, {{"num_splits", 3}});
    index_Split->set_output_size(3);
    auto index_Reshape = wrap_type<ov::op::v1::Reshape>({unsqueeze_Unsqueeze_1, pattern::any_input()}, {{"special_zero", true}});

    auto expert_scatter = mlp3_no_bias_swiglu_block(permute_Transpose, unsqueeze_Unsqueeze, index_Split->output(1), index_Reshape);
    
    // Pattern to match the operations after all experts - this captures the final scatter operation
    // that accumulates results from all experts, followed by reshape and residual add  
    // auto final_scatter = wrap_type<ov::op::v12::ScatterElementsUpdate>();
    auto original_shape = pattern::any_input();
    auto last_reshape = wrap_type<ov::op::v1::Reshape>({expert_scatter, original_shape}, {{"special_zero", false}});
    auto residual_input = pattern::any_input();
    auto last_add = wrap_type<ov::op::v1::Add>({residual_input, last_reshape}, {{"auto_broadcast", "numpy"}});

    auto callback = [=](const std::unordered_map<std::shared_ptr<Node>, std::vector<PatternValueMap>>& matches) {
        std::vector<std::shared_ptr<ov::pass::pattern::op::Block>> expert_nodes;
        std::vector<std::shared_ptr<Node>> weights_gate;
        std::vector<std::shared_ptr<Node>> weights_up;
        std::vector<std::shared_ptr<Node>> weights_down;
        std::vector<size_t> expert_idx;
        std::vector<expert_data> experts;
        
        // Get the final operations from the pattern
        // auto final_scatter_node = matches.at(final_scatter)[0].at(final_scatter).get_node_shared_ptr();
        auto last_add_node = matches.at(last_add)[0].at(last_add).get_node_shared_ptr();
        auto last_reshape_node = matches.at(last_add)[0].at(last_reshape).get_node_shared_ptr();
        // auto last_add_node = matches.at(last_add)[0].at(last_add).get_node_shared_ptr();
        // auto original_shape_node = matches.at(original_shape)[0].at(original_shape).get_node_shared_ptr();
        // auto residual_input_node = matches.at(residual_input)[0].at(residual_input).get_node_shared_ptr();
        
        for (const auto& pm : matches.at(expert_scatter)) {
            auto block_node = ov::as_type_ptr<ov::pass::pattern::op::Block>(pm.at(expert_scatter).get_node_shared_ptr());
            auto gate_proj_node = expert_scatter->get_anchor("gate_proj_weight", pm).value().get_node_shared_ptr();
            auto up_proj_node = expert_scatter->get_anchor("up_proj_weight", pm).value().get_node_shared_ptr();
            auto down_proj_node = expert_scatter->get_anchor("down_proj_weight", pm).value().get_node_shared_ptr();
            auto expert_id_node = expert_scatter->get_anchor("expert_id", pm).value().get_node_shared_ptr();
            auto expert_id_const = ov::as_type_ptr<ov::op::v0::Constant>(expert_id_node);
            experts.push_back({block_node, gate_proj_node, up_proj_node, down_proj_node, expert_id_const->cast_vector<size_t>()[0]});
        }
        std::sort(experts.begin(), experts.end(), [](const expert_data& a, const expert_data& b) {
            return a.expert_id < b.expert_id;
        });
        auto const_0 = ov::op::v0::Constant::create(element::i64, Shape{1}, {0});
        for (expert_data& expert : experts) {
            // Prepare weights for concatenation by unsqueezing 0 dimension
            weights_gate.push_back(ov::op::util::make_try_fold<ov::op::v0::Unsqueeze>(expert.gate_proj_weight, const_0));
            weights_up.push_back(ov::op::util::make_try_fold<ov::op::v0::Unsqueeze>(expert.up_proj_weight, const_0));
            weights_down.push_back(ov::op::util::make_try_fold<ov::op::v0::Unsqueeze>(expert.down_proj_weight, const_0));
        }

        auto fused_gate_weights = ov::op::util::make_try_fold<ov::op::v0::Concat>(weights_gate, 0);
        auto fused_up_weights = ov::op::util::make_try_fold<ov::op::v0::Concat>(weights_up, 0);
        auto fused_down_weights = ov::op::util::make_try_fold<ov::op::v0::Concat>(weights_down, 0);

        auto last_scatter_node = experts.back().expert_block->outputs()[0];
        
        // Debug output for the collected operations
        std::cout << "Collected " << experts.size() << " experts" << std::endl;
        std::cout << "Last reshape node: " << last_reshape_node->get_friendly_name() << std::endl;
        std::cout << "Last add node: " << last_add_node->get_friendly_name() << std::endl;
        
        // Get some key nodes from the original pattern to understand structure
        if (experts.empty()) {
            std::cout << "No experts found!" << std::endl;
            return false;
        }
        
        // Get input from the pattern - this should be the post-attention layernorm output
        auto view_reshape_node = matches.at(last_add)[0].at(view_Reshape).get_node_shared_ptr();
        auto original_shape_node = matches.at(last_add)[0].at(original_shape).get_node_shared_ptr();
        auto residual_input_node = matches.at(last_add)[0].at(residual_input).get_node_shared_ptr();
        
        // Find the shape constant - should be [-1, 64]
        auto shape_const_node = shape_const;
        
        // Extract dynamic values from the collected experts
        size_t num_experts = experts.size();
        
        // Use OpenVINO ops to extract dimensions dynamically
        auto gate_weight_shape_op = std::make_shared<ov::op::v3::ShapeOf>(experts[0].gate_proj_weight, element::i64);
        auto hidden_dim_scalar = std::make_shared<ov::op::v8::Gather>(gate_weight_shape_op, 
                                                                      ov::op::v0::Constant::create(element::i64, {}, {1}), 
                                                                      ov::op::v0::Constant::create(element::i64, {}, {0}));
        auto hidden_dim = std::make_shared<ov::op::v0::Unsqueeze>(hidden_dim_scalar, ov::op::v0::Constant::create(element::i64, {}, {0}));
        auto intermediate_dim = std::make_shared<ov::op::v8::Gather>(gate_weight_shape_op, 
                                                                     ov::op::v0::Constant::create(element::i64, {}, {0}), 
                                                                     ov::op::v0::Constant::create(element::i64, {}, {0}));
        
        std::cout << "Number of experts: " << num_experts << std::endl;
        
        // Create the fused MoE computation following the target pattern
        // 1. Tile the input to replicate for all experts
        auto num_experts_const = ov::op::v0::Constant::create(element::i64, {}, {static_cast<int64_t>(num_experts)});
        auto tile_shape = ov::op::v0::Constant::create(element::i64, {2}, {static_cast<int64_t>(num_experts), static_cast<int64_t>(1)});
        auto repeated_input = std::make_shared<ov::op::v0::Tile>(view_reshape_node, tile_shape);
        
        // 2. Reshape to [num_experts, -1, hidden_dim] for batched computation  
        auto batched_shape = std::make_shared<ov::op::v0::Concat>(
            OutputVector{std::make_shared<ov::op::v0::Unsqueeze>(num_experts_const, ov::op::v0::Constant::create(element::i64, {}, {0})), 
                         ov::op::v0::Constant::create(element::i64, {1}, {-1}), 
                         hidden_dim}, 0);
        auto batched_input = std::make_shared<ov::op::v1::Reshape>(repeated_input, batched_shape, true);
        
        // 3. Perform batched matrix multiplications with fused weights
        auto gate_bmm = std::make_shared<ov::op::v0::MatMul>(batched_input, fused_gate_weights, false, true);
        auto gate_swish = std::make_shared<ov::op::v4::Swish>(gate_bmm);
        
        auto up_bmm = std::make_shared<ov::op::v0::MatMul>(batched_input, fused_up_weights, false, true);
        auto swiglu_mul = std::make_shared<ov::op::v1::Multiply>(gate_swish, up_bmm);
        
        auto down_bmm = std::make_shared<ov::op::v0::MatMul>(swiglu_mul, fused_down_weights, false, true);
        
        // 4. Create shape constants for proper reshaping back
        auto batch_shape = std::make_shared<ov::op::v3::ShapeOf>(view_reshape_node, element::i64);
        auto batch_size = std::make_shared<ov::op::v8::Gather>(batch_shape, 
                                                               ov::op::v0::Constant::create(element::i64, {}, {0}), 
                                                               ov::op::v0::Constant::create(element::i64, {}, {0}));
        auto seq_len = std::make_shared<ov::op::v8::Gather>(original_shape_node, 
                                                            ov::op::v0::Constant::create(element::i64, {}, {1}), 
                                                            ov::op::v0::Constant::create(element::i64, {}, {0}));
        
        // Create shape [num_experts, batch_size, seq_len, hidden_dim]
        auto expert_output_shape = std::make_shared<ov::op::v0::Concat>(
            OutputVector{std::make_shared<ov::op::v0::Unsqueeze>(num_experts_const, ov::op::v0::Constant::create(element::i64, {}, {0})), 
                         std::make_shared<ov::op::v0::Unsqueeze>(batch_size, ov::op::v0::Constant::create(element::i64, {}, {0})), 
                         std::make_shared<ov::op::v0::Unsqueeze>(seq_len, ov::op::v0::Constant::create(element::i64, {}, {0})), 
                         hidden_dim}, 0);
        
        auto expert_outputs = std::make_shared<ov::op::v1::Reshape>(down_bmm, expert_output_shape, false);
        
        // 5. Extract router information from existing pattern
        // Get the TopK node from the original pattern instead of recreating it
        auto topk = matches.at(last_add)[0].at(topk_TopK).get_node_shared_ptr();
        // auto topk = topk_TopK;
        // if (!topk) {
        //     std::cout << "Failed to get TopK node from pattern" << std::endl;
        //     return false;
        // }
        
        // 6. Create routing weights tensor and apply to experts
        auto routing_shape = std::make_shared<ov::op::v0::Concat>(
            OutputVector{std::make_shared<ov::op::v0::Unsqueeze>(num_experts_const, ov::op::v0::Constant::create(element::i64, {}, {0})), 
                         std::make_shared<ov::op::v0::Unsqueeze>(batch_size, ov::op::v0::Constant::create(element::i64, {}, {0})), 
                         std::make_shared<ov::op::v0::Unsqueeze>(seq_len, ov::op::v0::Constant::create(element::i64, {}, {0}))}, 0);
        
        auto routing_weights = std::make_shared<ov::op::v1::Reshape>(topk->output(0), routing_shape, false);
        auto routing_weights_4d = std::make_shared<ov::op::v0::Unsqueeze>(routing_weights, ov::op::v0::Constant::create(element::i64, {}, {3}));
        
        // 7. Apply routing weights and sum across experts  
        auto weighted_outputs = std::make_shared<ov::op::v1::Multiply>(expert_outputs, routing_weights_4d);
        auto final_output = std::make_shared<ov::op::v1::ReduceSum>(weighted_outputs, 
                                                                     ov::op::v0::Constant::create(element::i64, {}, {0}), 
                                                                     false);
        
        // 8. Reshape back to original shape and add residual connection
        auto final_reshape = std::make_shared<ov::op::v1::Reshape>(final_output, original_shape_node, false);
        auto final_add = std::make_shared<ov::op::v1::Add>(residual_input_node, final_reshape);
        
        // Replace the last add node with our new computation
        ov::replace_node(last_add_node, final_add);
        
        return true;
        // auto reshape_after_scatter = ov::as_type_ptr<ov::op::v1::Reshape>(last_scatter_node->);

        // auto repeat_Tile = makeOP<opset1::Tile>({reshape_Reshape, {8,1}});
        // auto view_Reshape = makeOP<opset1::Reshape>({repeat_Tile, {8,-1,64}}, {{"special_zero", true}});
        // auto bmm_MatMul = makeOP<opset1::MatMul>({view_Reshape, fused_gate_weights}, {{"transpose_a", false}, {"transpose_b", true}});
        // auto silu_Swish = makeOP<opset4::Swish>({bmm_MatMul});
        // auto bmm_MatMul_1 = makeOP<opset1::MatMul>({view_Reshape, fused_up_weights}, {{"transpose_a", false}, {"transpose_b", true}});
        // auto mul_Multiply = makeOP<opset1::Multiply>({silu_Swish, bmm_MatMul_1}, {{"auto_broadcast", "numpy"}});
        // auto bmm_MatMul_2 = makeOP<opset1::MatMul>({mul_Multiply, fused_down_weights}, {{"transpose_a", false}, {"transpose_b", true}});
        // auto size_Gather = makeOP<opset8::Gather>({ShapeOf_14089, {2}, 0}, {{"batch_dims", 0}});
        // auto ListConstruct_5 = makeOP<opset1::Concat>({{8}, Reshape_14179, {-1}, size_Gather}, {{"axis", 0}});
        // auto view_Reshape_1 = makeOP<opset1::Reshape>({bmm_MatMul_2, ListConstruct_5}, {{"special_zero", false}});
        return true;
    };

    register_patterns({expert_scatter, last_add}, callback, true);
}

// bool ov::pass::FuseMOE::run_on_model(const std::shared_ptr<ov::Model>& model) {
//     RUN_ON_MODEL_SCOPE(FuseMOE);
//     ov::pass::Manager manager(get_pass_config(), "FuseMOE");

//     // manager.register_pass<ov::pass::EliminateSqueeze>();
//     // Use the unified FuseMOE transformation
//     manager.register_pass<ov::pass::PrintModel>("before_fuse_moe_pseudocode.cpp");
//     manager.register_pass<ov::pass::FuseMOEUnified>();

//     manager.run_passes(model);
//     return false;
// }

}  // namespace pass
}  // namespace ov
