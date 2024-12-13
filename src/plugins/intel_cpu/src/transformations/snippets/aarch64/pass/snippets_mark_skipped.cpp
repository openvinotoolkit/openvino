// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "snippets_mark_skipped.hpp"

#include "snippets/pass/tokenization.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/utils/utils.hpp"

#include "transformations/utils/utils.hpp"
#include "transformations/utils.hpp"
#include "utils/general_utils.h"
#include "utils/cpu_utils.hpp"

#include "itt.hpp"


namespace ov {
namespace intel_cpu {

namespace {
static const int DEFAULT_AXIS = 1;
NodeFusingType GetNodeFusingType(const std::shared_ptr<const Node> &node) {
    auto &rt = node->get_rt_info();
    const auto rinfo = rt.find("MayBeFusedInPlugin");
    if (rinfo == rt.end())
        return NodeFusingType::NotSet;
    return rinfo->second.as<NodeFusingType>();
}
void SetNodeFusingType(const std::shared_ptr<Node> &node, NodeFusingType nodeType) {
    auto &rt = node->get_rt_info();
    rt["MayBeFusedInPlugin"] = nodeType;
}
std::vector<NodeFusingType> getContinuableChains(const std::shared_ptr<const Node> &node) {
    std::vector<NodeFusingType> result;
    for (const auto& input : node->inputs()) {
        const auto parent = input.get_source_output().get_node_shared_ptr();
        const auto snt = GetNodeFusingType(parent);
        if (snt > NodeFusingType::FusedTerminator) {
            result.push_back(snt);
        }
    }
    return result;
}
int getNumNonConstInputs(const std::shared_ptr<const Node> &node) {
    int num_non_const_inputs = 0;
    for (const auto &parent_out : node->input_values()) {
        const auto parent = parent_out.get_node_shared_ptr();
        if (ov::is_type<ov::op::v1::Reshape>(parent)) {
            for (const auto &grandparent_out : parent->input_values()) {
                const auto grandparent = grandparent_out.get_node_shared_ptr();
                if (!ov::is_type<ov::op::v0::Constant>(grandparent))
                    num_non_const_inputs++;
            }
        } else if (!ov::is_type<ov::op::v0::Constant>(parent)) {
            num_non_const_inputs++;
        }
    }
    return num_non_const_inputs;
}
bool isFullyConnected(const std::shared_ptr<const ov::Node>& node) {
    if (!ov::is_type<ov::op::v0::MatMul>(node))
        return false;
    const auto out_activations = node->input_value(0);
    const auto out_weights = node->input_value(1);
    const auto rank_a = out_activations.get_partial_shape().rank();
    const auto rank_w = out_weights.get_partial_shape().rank();
    return out_weights.get_partial_shape().is_static() &&
           rank_a.is_static() && rank_w.is_static() &&
           rank_a.get_length() != 1 && rank_w.get_length() != 1 &&
           rank_w.get_length() <= 3 &&
           ov::op::util::is_on_constant_path(out_weights);
}

bool SupportsFusingWithConvolution_Simple(const std::shared_ptr<const Node> &node) {
    // Note: some other operations support this fusing (SoftPlus, Sqrt).
    // Skip them here, when they are supported by Snippets ARM. Ticket: 141170.
    return ov::is_type<ov::op::v0::Abs>(node) ||
           ov::is_type<ov::op::v0::Clamp>(node) ||
           ov::is_type<ov::op::v0::Elu>(node) ||
           ov::is_type<ov::op::v0::Relu>(node) ||
           ov::is_type<ov::op::v0::Sigmoid>(node) ||
           ov::is_type<ov::op::v0::Tanh>(node);
}
// Convolution is a special case, since it supports peculiar fusings
bool isSuitableConvolutionParent(const std::shared_ptr<const Node> &node) {
    const bool is_suitable_node = ov::is_type<ov::op::v1::Convolution>(node) ||
                                  ov::is_type<ov::op::v1::GroupConvolution>(node);
    // has a single output, connected to a single child
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_suitable_node && has_only_child;
}
bool isSuitableBinaryConvolutionParent(const std::shared_ptr<const Node> &node) {
    const bool is_suitable_node = ov::is_type<ov::op::v1::BinaryConvolution>(node);
    // has a single output, connected to a single child
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_suitable_node && has_only_child;
}
bool isSuitableMiscParent(const std::shared_ptr<const Node> &node) {
    const bool is_suitable_node = ov::is_type<ov::op::v0::NormalizeL2>(node) ||
                                  ov::is_type<ov::op::v1::ConvolutionBackpropData>(node) ||
                                  ov::is_type<ov::op::v1::GroupConvolutionBackpropData>(node);
    // has a single output, connected to a single child
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_suitable_node && has_only_child;
}
// Matmul is a special case, since it supports simple + bias fusings
bool isSuitableMatMulParent(const std::shared_ptr<const Node> &node) {
    const bool is_suitable_node = ov::is_type<ov::op::v0::MatMul>(node);
    // has a single output, connected to a single child
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_suitable_node && has_only_child;
}
bool isSuitablePoolChild(const std::shared_ptr<const Node> &node) {
    const bool is_suitable_node = ov::is_type<ov::op::v1::MaxPool>(node);
    // has a single output, connected to a single child
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_suitable_node && has_only_child;
}
bool isSuitableChildForFusingSimple(const std::shared_ptr<const Node> &node) {
    // Note: Fusing child is allowed to have several users, but that must be the end of the chain
    return SupportsFusingWithConvolution_Simple(node) && getNumNonConstInputs(node) == 1;
}
bool isSuitableChildForFusingBias(const std::shared_ptr<const Node> &node, int fusingAxis) {
    if (!ov::is_type<ov::op::v1::Add>(node))
        return false;

    auto is_suitable_parent = [](const std::shared_ptr<const Node> &node) {
        return  (ov::is_type<ov::op::v1::Convolution>(node) ||
                 ov::is_type<ov::op::v1::GroupConvolution>(node) ||
                 ov::is_type<ov::op::v0::MatMul>(node));
    };

    for (const auto &in : node->inputs()) {
        const auto& parent_out = in.get_source_output();
        const auto& parent = parent_out.get_node_shared_ptr();
        const auto& parent_pshape = parent_out.get_partial_shape();
        if (is_suitable_parent(parent) && parent_pshape.rank().is_static()) {
            if (parent->get_output_target_inputs(0).size() > 1)
                break;
            const auto bias_port = 1 - in.get_index();
            const auto bias_out = node->input_value(bias_port);
            if ((bias_out.get_target_inputs().size() > 1) || !ov::op::util::is_on_constant_path(bias_out))
                break;
            const auto& bias_pshape = bias_out.get_partial_shape();
            if (bias_pshape.is_dynamic())
                break;
            const auto bias_shape_norm = getNormalizedDimsBySize(bias_pshape.get_shape(), parent_pshape.size());
            if (fusingAxis >= static_cast<int>(bias_shape_norm.size()) || fusingAxis >= static_cast<int>(parent_pshape.size()) ||
                bias_shape_norm.size() != parent_pshape.size() || bias_shape_norm.size() < 2)
                break;
            if (parent_pshape[fusingAxis].is_dynamic())
                break;
            if ((bias_shape_norm[fusingAxis] == static_cast<size_t>(parent_pshape[fusingAxis].get_length())) &&
                (bias_shape_norm[fusingAxis] == shape_size(bias_shape_norm)))
                return true;
        }
    }
    return false;
}
// Continue fusing chain of the passed type if the node has one child
// Otherwise mark node as FusedTerminator (Fused, but fusing chain is interrupted)
void PropagateIfHasOnlyChild(const std::shared_ptr<Node> &node, NodeFusingType nodeType) {
    const auto out = node->outputs();
    const bool has_only_child = out.size() == 1 && out[0].get_target_inputs().size() == 1;
    SetNodeFusingType(node, has_only_child ? nodeType : NodeFusingType::FusedTerminator);
}
// todo: Skipping MultiSubGraphOp such as TensorIterator, Loop and If. Snippets might tokenize their bodies in the future.
//  Note that the function is recurrent, since there might be multi-level MultiSubGraphOp, if(){if(){}}else{} for example.
void MarkSubgraphOpAsSkipped(const std::shared_ptr<Node> &node) {
    if (ov::is_type<ov::op::util::MultiSubGraphOp>(node)) {
        std::vector<std::shared_ptr<ov::Model>> models{};
        // Covers TensorIterator and Loop
        if (auto s = ov::as_type_ptr<ov::op::util::SubGraphOp>(node)) {
            models.push_back(s->get_function());
        // Add new multi-body subgraph op here
        } else if (auto if_op  = ov::as_type_ptr<ov::op::v8::If>(node)) {
            models.push_back(if_op->get_then_body());
            models.push_back(if_op->get_else_body());
        }
        for (auto& m : models) {
            for (auto& n : m->get_ops()) {
                snippets::pass::SetSnippetsNodeType(n, snippets::pass::SnippetsNodeType::SkippedByPlugin);
                MarkSubgraphOpAsSkipped(n);
            }
        }
    }
}

bool isSuitableConvert(const std::shared_ptr<const Node>& node) {
    if (!ov::is_type<ov::op::v0::Convert>(node))
        return false;
    auto isSuitableParent = [](const std::shared_ptr<const Node>& node) {
        for (const auto& input : node->inputs()) {
            const auto parent = input.get_source_output().get_node_shared_ptr();
            if (!ov::is_type<ov::op::v3::ReadValue>(parent))
                return false;
        }
        return true;
    };
    auto isSuitableChild = [](const std::shared_ptr<const Node>& node) {
        for (const auto &out : node->outputs()) {
            const auto &child = out.get_node_shared_ptr();
            if (!ov::is_type<ov::op::v3::Assign>(child))
                return false;
        }
        return true;
    };
    return isSuitableParent(node) || isSuitableChild(node);
}

auto is_skipped_op(const std::shared_ptr<ov::Node>& op) -> bool {
    return ov::is_type<ov::op::v0::Constant>(op) ||
           ov::is_type<ov::op::v0::Parameter>(op) ||
           ov::is_type<ov::op::v0::Result>(op);
}
} // namespace

bool SnippetsMarkSkipped::run_on_model(const std::shared_ptr<ov::Model> &m) {
    RUN_ON_MODEL_SCOPE(SnippetsMarkSkipped);
    int channelAxis = DEFAULT_AXIS;
    for (auto &node : m->get_ordered_ops()) {
        if (is_skipped_op(node))
            continue;
        if (isSuitableConvolutionParent(node)) {
            // Initiate fusing chain
            SetNodeFusingType(node, NodeFusingType::FusedWithConvolution);
            channelAxis = DEFAULT_AXIS;
        } else if (isSuitableBinaryConvolutionParent(node)) {
            SetNodeFusingType(node, NodeFusingType::FusedWithBinaryConvolution);
            channelAxis = DEFAULT_AXIS;
        } else if (isSuitableMiscParent(node)) {
            channelAxis = DEFAULT_AXIS;
            SetNodeFusingType(node, NodeFusingType::FusedWithMisc);
        } else if (isSuitableMatMulParent(node)) {
            const bool is_fc = isFullyConnected(node);
            const auto out_rank = node->get_output_partial_shape(0).rank();
            if (is_fc) {
                SetNodeFusingType(node, NodeFusingType::FusedWithFC);
                channelAxis = out_rank.is_static() ? (out_rank.get_length() == 3 ? 2 : 1) : DEFAULT_AXIS;
            } else {
                SetNodeFusingType(node, NodeFusingType::FusedWithMatMul);
                channelAxis = out_rank.is_static() ? out_rank.get_length() - 1 : DEFAULT_AXIS;
            }
        } else if (isSuitableConvert(node)) {
            SetSnippetsNodeType(node, snippets::pass::SnippetsNodeType::SkippedByPlugin);
            channelAxis = DEFAULT_AXIS;
        } else {
            for (const auto fusingChainType : getContinuableChains(node)) {
                if (isSuitableChildForFusingBias(node, channelAxis)) {
                    PropagateIfHasOnlyChild(node, fusingChainType);
                } else if (isSuitableChildForFusingSimple(node)) {
#if defined (OV_CPU_WITH_ACL)
                    if (one_of(fusingChainType, NodeFusingType::FusedWithConvolution, NodeFusingType::FusedWithBinaryConvolution)) {
                        PropagateIfHasOnlyChild(node, NodeFusingType::FusedTerminator);
                        continue;
                    }
#endif
                    PropagateIfHasOnlyChild(node, fusingChainType);
                } else if (one_of(fusingChainType, NodeFusingType::FusedWithConvolution, NodeFusingType::FusedWithBinaryConvolution)) {
                    if (isSuitablePoolChild(node)) {
                        PropagateIfHasOnlyChild(node, fusingChainType);
                    }
                }
            }
        }

        if (GetNodeFusingType(node) != NodeFusingType::NotSet) {
            SetSnippetsNodeType(node, snippets::pass::SnippetsNodeType::SkippedByPlugin);
        } else {
            MarkSubgraphOpAsSkipped(node);
        }
    }
    return true;
}

}   // namespace intel_cpu
}   // namespace ov
