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

namespace ov {
namespace pass {

using namespace ov::gen_pattern;
using namespace ov::pass;
using namespace ov::pass::pattern;

namespace {
std::shared_ptr<Node> mlp3_no_bias_swiglu_block(const Output<Node>& permute_Transpose, // Transpose -> OneHot -> TopK -> Softmax -> MatMul -> Hidden States
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
    // auto linear_MatMul = wrap_type<ov::op::v0::MatMul>({view_Reshape, self_model_layers_1_mlp_gate_weight}, {{"transpose_a", false}, {"transpose_b", true}});
    auto linear_MatMul = pattern::any_input();
    auto softmax_Softmax = wrap_type<ov::op::v8::Softmax>({linear_MatMul}, {{"axis", 1}});
    auto topk_TopK = wrap_type<ov::op::v11::TopK>({softmax_Softmax, 2}, {{"axis", -1}, {"mode", "max"}, {"sort", "value"}, {"index_element_type", "i64"}, {"stable", false}});
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
    auto callback = [=](const std::unordered_map<std::shared_ptr<Node>, std::vector<PatternValueMap>>& matches) {
        // auto match = matches.at(expert_scatter);
        std::unordered_set<Node*> expert_nodes;
        std::unordered_map<Node*, const PatternValueMap*> node_to_expert;
        std::vector<Node*> weights_gate;
        std::vector<Node*> weights_up;
        std::vector<Node*> weights_down;
        std::vector<size_t> expert_idx;
        for (const auto& pm : matches.at(expert_scatter)) {
            auto root = pm.at(expert_scatter).get_node();
            auto block = std::dynamic_pointer_cast<ov::pass::pattern::op::Block>(pm.at(expert_scatter).get_node_shared_ptr());
            weights_gate.push_back(block->get_anchor("gate_proj_weight", pm).value().get_node());
            weights_up.push_back(block->get_anchor("up_proj_weight", pm).value().get_node());
            weights_down.push_back(block->get_anchor("down_proj_weight", pm).value().get_node());
            auto expert_id_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(block->get_anchor("expert_id", pm).value().get_node_shared_ptr());
            expert_idx.push_back(expert_id_const->cast_vector<size_t>()[0]);
            expert_nodes.insert(root);
            node_to_expert[root] = &pm;
        }
        auto num_experts = node_to_expert.size();
        // TODO
        // 1. Collect expert weights
        // 2. If needed, sort them based on expert id from gather
        // 3. Cat corresponding expert dimensions together. Try constfold using make_try_fold?
        // 4. Update target subgraph, connect inputs/outputs
        // 5. Replace subgraph
        return true;
    };

    register_patterns({expert_scatter}, callback, true);
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
