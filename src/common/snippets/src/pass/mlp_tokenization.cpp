// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/mlp_tokenization.hpp"
#include "snippets/pass/collapse_subgraph.hpp"

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/opsets/opset4.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils/tokenization_utils.hpp"

ov::snippets::pass::TokenizeMLPSnippets::TokenizeMLPSnippets(const SnippetsTokenization::Config& config) {
    MATCHER_SCOPE(TokenizeMLPSnippets);
    using namespace ov::pass::pattern;
    auto constant = wrap_type<ov::op::v0::Constant>();
    // todo: check transpose_b = true
    // auto fc_matmul = ov::pass::pattern::wrap_type<ov::opset1::MatMul>({any_input(), constant});

    ov::pass::pattern::op::NodePredicate fc_pred = [](const std::shared_ptr<ov::Node>& n){
        if (auto mm = ov::as_type_ptr<ov::opset1::MatMul>(n)) {
            const bool has_const_weights = ov::is_type<opset1::Constant>(mm->get_input_node_shared_ptr(1));
            const bool is_transposed = mm->get_transpose_b();
            return has_const_weights && is_transposed;
//            return ov::is_type<opset1::Constant>(mm->get_input_node_shared_ptr(1)) &&
//                   mm->get_transpose_b();
        }
        return false;
    };
    auto fc_matmul = std::make_shared<ov::pass::pattern::op::Label>(any_input(), fc_pred);

    auto callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::TokenizeMLPSnippets")
        const auto last_matmul = m.get_match_root();
        if (transformation_callback(last_matmul)) {
            return false;
        }
        NodeVector ordered_ops {last_matmul};
        auto fuse_single_in_chain = [&ordered_ops](const std::shared_ptr<ov::Node>& start) {
            auto in = start;
            while (TokenizeSnippets::AppropriateForSubgraph(in) && in->get_input_size() == 1) {
                ordered_ops.push_back(in);
                in = in->get_input_node_shared_ptr(0);
            }
            return in;
        };
        const auto last_not_fused = fuse_single_in_chain(last_matmul->get_input_node_shared_ptr(0));
        // Next node to be fused must have 2 inputs. Abort if not true
        if (!TokenizeSnippets::AppropriateForSubgraph(last_not_fused) || last_not_fused->get_input_size() != 2)
            return false;
        ordered_ops.push_back(last_not_fused);

        const auto left_matmul = fuse_single_in_chain(last_not_fused->get_input_node_shared_ptr(0));
        const auto right_matmul = fuse_single_in_chain(last_not_fused->get_input_node_shared_ptr(1));
        // Eltwise fusing chains must be interrupted by FullyConnected nodes
        auto match_allowed_by_callback = [&m, this](const std::shared_ptr<Node>& n) {
            return m.match(n) && !transformation_callback(m.get_match_root());
        };
        if (!match_allowed_by_callback(left_matmul) || !match_allowed_by_callback(right_matmul))
            return false;
        ordered_ops.push_back(right_matmul);
        ordered_ops.push_back(left_matmul);
        // Note: Insert Reshape between Constant and MatMul if the rank of Constant's tensor lower than the rank of activations
        auto reshape_constant = [](const std::shared_ptr<Node>& n) {
            const auto& act_rank = n->get_input_partial_shape(0).rank();
            const auto& wei_shape = n->get_input_shape(1);
            OPENVINO_ASSERT(act_rank.is_static(), "Rank of input shapes should be static at this point");
            const auto act_rank_len = static_cast<size_t>(act_rank.get_length());
            const auto wei_rank_len = wei_shape.size();
            OPENVINO_ASSERT(wei_rank_len <= act_rank_len, "Weights rank is expected to be smaller than the activation's rank");
            if (act_rank_len == wei_rank_len)
                return;
            ov::Shape target_shape(act_rank_len, 1);
            std::copy(wei_shape.rbegin(), wei_shape.rend(), target_shape.rbegin());
            const auto shape_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{target_shape.size()}, target_shape);
            const auto reshape = std::make_shared<ov::op::v1::Reshape>(n->get_input_source_output(1), shape_const, false);
            n->input(1).replace_source_output(reshape->output(0));
        };
        reshape_constant(last_matmul);
        reshape_constant(left_matmul);
        reshape_constant(right_matmul);
        std::reverse(ordered_ops.begin(), ordered_ops.end());

        auto subgraph = utils::wrap_nodes_as_subgraph(ordered_ops);
        // todo: seems like we don't need to set this thing if subgraph is already complete
        //subgraph->set_virtual_port_count(hidden_virtual_ports_count);
        SetSnippetsSubgraphType(subgraph, SnippetsSubgraphType::Completed);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(fc_matmul, matcher_name);
    register_matcher(matcher, callback);
}
