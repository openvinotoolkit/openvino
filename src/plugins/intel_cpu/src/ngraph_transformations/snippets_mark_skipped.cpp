// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "snippets_mark_skipped.hpp"
#include <snippets/pass/collapse_subgraph.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <utils/general_utils.h>
#include <utils/cpu_utils.hpp>

#include "itt.hpp"

using namespace ngraph;

namespace ov {
namespace intel_cpu {

namespace {
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
        if (ov::is_type<ngraph::op::v1::Reshape>(parent)) {
            for (const auto &grandparent_out : parent->input_values()) {
                const auto grandparent = grandparent_out.get_node_shared_ptr();
                if (!ngraph::op::is_constant(grandparent))
                    num_non_const_inputs++;
            }
        } else if (!ngraph::op::is_constant(parent)) {
            num_non_const_inputs++;
        }
    }
    return num_non_const_inputs;
}
bool SupportsFusingWithConvolution_SumActivation(const std::shared_ptr<const Node> &node) {
    // todo: Do all PReLUs are fused? Not sure about round and softRelu
    // EltwiseRoundHalfToEven, EltwiseRoundHalfAwayFromZero, EltwiseSoftRelu
    return  ov::is_type<ngraph::op::Relu>(node) ||
            ov::is_type<ngraph::op::PRelu>(node) ||
            ov::is_type<ngraph::op::Elu>(node) ||
            ov::is_type<ngraph::op::Sigmoid>(node) ||
            ov::is_type<ngraph::op::v5::HSigmoid>(node) ||
            ov::is_type<ngraph::op::Clamp>(node) ||
            ov::is_type<ngraph::op::v4::Swish>(node) ||
            ov::is_type<ngraph::op::v4::HSwish>(node) ||
            ov::is_type<ngraph::op::v4::Mish>(node) ||
            ov::is_type<ngraph::op::v5::Round>(node);
}

bool canBePerformedAsScaleShift(const std::shared_ptr<const Node> &node, const int channelAxis) {
    size_t fusingPort = 0;
    size_t numNonConstInputs = 0;
    ov::PartialShape dataShape;
    for (size_t i = 0; i < node->get_input_size(); i++) {
        const auto parent = node->get_input_node_shared_ptr(i);
        if (!ngraph::op::is_constant(parent)) {
            fusingPort = i;
            dataShape = node->get_input_partial_shape(i);
            // only one non-const parent is allowed
            if (dataShape.is_dynamic() || ++numNonConstInputs != 1)
                return false;
        } else {
            // every const parent must have exactly one child
            const auto out = parent->outputs();
            const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
            if (!has_only_child)
                return false;
        }
    }

    const auto isBroadcastableToDataInput = [&]() {
        for (size_t i = 0; i < node->get_input_size(); i++) {
            if (i == fusingPort)
                continue;
            const ov::PartialShape weightShape = node->get_input_partial_shape(i);
            if (weightShape.is_dynamic() ||
                !isPerTensorOrPerChannelBroadcastable(dataShape.get_shape(), weightShape.get_shape(), channelAxis, true))
                return false;
        }
        return true;
    };

    // Prelu and MulAdd are still ignored
    // isConvertablePowerStatic() is ignored
    return (ov::is_type<ngraph::opset1::Add>(node) ||
            ov::is_type<ngraph::opset1::Multiply>(node) ||
            ov::is_type<ngraph::opset1::Subtract>(node) ||
            ov::is_type<ngraph::opset1::Divide>(node)) &&
           isBroadcastableToDataInput();
}

bool SupportsFusingWithConvolution_Simple(const std::shared_ptr<const Node> &node, const int channelAxis = 1) {
    return SupportsFusingWithConvolution_SumActivation(node) ||
           ov::is_type<ngraph::op::Tanh>(node) ||
           ov::is_type<ngraph::op::v0::Gelu>(node) ||
           ov::is_type<ngraph::op::v7::Gelu>(node) ||
           ov::is_type<ngraph::op::Abs>(node) ||
           ov::is_type<ngraph::op::Sqrt>(node) ||
           canBePerformedAsScaleShift(node, channelAxis);
}
// Convolution is a special case, since it supports peculiar fusings
bool isSuitableConvolutionParent(const std::shared_ptr<const Node> &node) {
    const bool is_suitable_node = ov::is_type<ngraph::op::v1::Convolution>(node) ||
                                  ov::is_type<ngraph::op::v1::GroupConvolution>(node);
    // has a single output, connected to a single child
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_suitable_node && has_only_child;
}
bool isSuitableBinaryConvolutionParent(const std::shared_ptr<const Node> &node) {
    const bool is_suitable_node = ov::is_type<ngraph::op::v1::BinaryConvolution>(node);
    // has a single output, connected to a single child
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_suitable_node && has_only_child;
}
int getChannelAxis(const ov::AxisSet &axes, bool keep_dims) {
    int channelAxis = 1;
    if (!keep_dims) {
        for (auto &axis : axes) {
            if (axis == 1) {
                // channel axis has been reduced and doesn't exist any more
                channelAxis = -1;
                break;
            } else if (axis == 0) {
                channelAxis = 0;
            }
        }
    }
    return channelAxis;
}
bool isSuitableMiscParent(const std::shared_ptr<const Node> &node, int &channelAxis) {
    const bool is_suitable_node = ov::is_type<ngraph::op::v0::MVN>(node) ||
                                  ov::is_type<ngraph::op::v6::MVN>(node) ||
                                  ov::is_type<ngraph::op::v0::NormalizeL2>(node) ||
                                  ov::is_type<ngraph::op::v0::Interpolate>(node) ||
                                  ov::is_type<ngraph::op::v4::Interpolate>(node) ||
                                  ov::is_type<ngraph::op::v0::LSTMCell>(node) ||
                                  ov::is_type<ngraph::op::v4::LSTMCell>(node) ||
                                  ov::is_type<ngraph::opset1::ConvolutionBackpropData>(node) ||
                                  ov::is_type<ngraph::op::util::ArithmeticReductionKeepDims>(node) ||
                                  ov::is_type<ngraph::op::util::LogicalReductionKeepDims>(node) ||
                                  ov::is_type<ngraph::opset1::GroupConvolutionBackpropData>(node);
    if (const auto reduce = std::dynamic_pointer_cast<const ngraph::op::util::ArithmeticReductionKeepDims>(node)) {
        channelAxis = getChannelAxis(reduce->get_reduction_axes(), reduce->get_keep_dims());
    } else if (const auto reduce = std::dynamic_pointer_cast<const ngraph::op::util::LogicalReductionKeepDims>(node)) {
        channelAxis = getChannelAxis(reduce->get_reduction_axes(), reduce->get_keep_dims());
    }
    // has a single output, connected to a single child
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_suitable_node && has_only_child;
}
// Matmul is a special case, since it supports simple + bias fusings
bool isSuitableMatMulParent(const std::shared_ptr<const Node> &node) {
    const bool is_suitable_node = ov::is_type<ngraph::op::MatMul>(node);
    // has a single output, connected to a single child
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_suitable_node && has_only_child;
}
bool isSuitablePoolChild(const std::shared_ptr<const Node> &node) {
    const bool is_suitable_node = ov::is_type<ngraph::op::v1::MaxPool>(node);
    // has a single output, connected to a single child
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_suitable_node && has_only_child;
}
bool isSuitableChildForFusingSimple(const std::shared_ptr<const Node> &node, int channelAxis = 1) {
    // Note: Fusing child is allowed to have several users, but that must be the end of the chain
    return SupportsFusingWithConvolution_Simple(node, channelAxis) && getNumNonConstInputs(node) == 1;
}
bool isSuitableChildForFusingMatMul(const std::shared_ptr<const Node> &node, NodeFusingType &updatedChainType) {
    int num_non_const_inputs = 0;
    bool can_be_converted_to_FC = false;
    ov::Shape bias_shape;
    ov::Shape matmul_shape;
    for (const auto &parent_out : node->input_values()) {
        const auto parent = parent_out.get_node_shared_ptr();
        if (ngraph::op::is_constant(parent)) {
            bias_shape = parent_out.get_shape();
            num_non_const_inputs++;
        } else {
            const auto pshape = parent_out.get_partial_shape();
            if (pshape.is_dynamic() || pshape.get_shape().empty())
                return false;
            matmul_shape = pshape.get_shape();
            const auto& grandparents = parent->input_values();
            // first check that weights are constant and both activations and weights have static shape
            if (grandparents.size() == 2 &&
                grandparents[0].get_partial_shape().is_static() &&
                grandparents[1].get_partial_shape().is_static() &&
                ov::is_type<ov::op::v0::Constant>(grandparents[1].get_node_shared_ptr())) {
                auto rank_a = grandparents[0].get_partial_shape().rank().get_length();
                auto rank_w = grandparents[1].get_partial_shape().rank().get_length();
                if (rank_a != 1 && rank_w != 1 && rank_a <= 3 && rank_w <= 3)
                    can_be_converted_to_FC = true;
            }
        }
    }
    if (num_non_const_inputs != 1)
        return false;

    // FuseMatMulAndSimpleOperation or FuseFullyConnectedAndSimpleOperation
    // Invoke SupportsFusingWithConvolution_Simple directly instead of isSuitableChildForFusingSimple to
    // eliminate getNumNonConstInputs() check
    int fusingAxis;
    if (can_be_converted_to_FC)
        fusingAxis = matmul_shape.size() == 3 ? 2 : 1;
    else
        fusingAxis = matmul_shape.size() - 1;
    if (SupportsFusingWithConvolution_Simple(node, fusingAxis)) {
        updatedChainType = NodeFusingType::FusedWithMisc;
        return true;
    }
    //    FullyConnectedBiasFusion
    if (!(can_be_converted_to_FC && ov::is_type<ngraph::opset1::Add>(node) &&
        bias_shape.back() == matmul_shape.back() &&
        bias_shape.back() == shape_size(bias_shape))) {
        return false;
    }
    // Fusing chain must be interrupted after the node, since reshape will be inserted
    if (bias_shape.size() >= 2)
        updatedChainType = NodeFusingType::FusedTerminator;
    return true;
}
bool isSuitableParentForFusingSumActivation(const std::shared_ptr<const Node> &node) {
    if (!ov::is_type<ngraph::op::v1::Add>(node))
        return false;
    auto isFusedBiasNode = [](std::shared_ptr<Node> n){
        if (!(ov::is_type<ngraph::op::v1::Add>(n) &&
              GetNodeFusingType(n) ==  NodeFusingType::FusedWithConvolution))
            return false;
        const auto conv = n->get_input_source_output(0);
        const auto bias = n->get_input_source_output(1);
        if (!(ngraph::op::is_constant(bias.get_node_shared_ptr()) && isSuitableConvolutionParent(conv.get_node_shared_ptr())))
            return false;
        const auto conv_shape = conv.get_partial_shape();
        const auto bias_shape = bias.get_partial_shape();
        if  (bias_shape.is_dynamic() || conv_shape.is_dynamic() || bias_shape.size() > conv_shape.size())
            return false;
        auto getNormalizedDims = [](const ov::Shape &dims, size_t ndims) -> std::vector<size_t>{
            std::vector<size_t> normalizedDims = dims;
            for (size_t i = 0; i < (ndims - dims.size()); i++) {
                normalizedDims.insert(normalizedDims.begin(), 1);
            }
            return normalizedDims;
        };
        const auto bias_norm_dims = getNormalizedDims(bias_shape.get_shape(), conv_shape.size());
        if (bias_norm_dims.size() < 2 || bias_norm_dims[0] != 1 || conv_shape[1] != bias_norm_dims[1])
            return false;
        for (size_t i = 2; i < bias_norm_dims.size(); i++) {
            if (bias_norm_dims[i] != 1)
                return false;
        }
        return true;
    };
    int num_conv_parents = 0;
    for (size_t i = 0; i < node->get_input_size(); i++) {
        const auto n = node->get_input_node_shared_ptr(i);
        //BinaryConvolution allows other ops to be fused before the Add, while Convolution doesn't
        num_conv_parents += (isSuitableConvolutionParent(n) || isFusedBiasNode(n) ||
                             GetNodeFusingType(n) == NodeFusingType::FusedWithBinaryConvolution);
    }
    return getNumNonConstInputs(node) == 2 && num_conv_parents >=1;
}
bool isSuitableChildForFusingSumActivation(const std::shared_ptr<const Node> &node) {
    return SupportsFusingWithConvolution_SumActivation(node);
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
} // namespace

bool SnippetsMarkSkipped::run_on_model(const std::shared_ptr<ov::Model> &m) {
    RUN_ON_MODEL_SCOPE(SnippetsMarkSkipped);
    int channelAxis = 1;
    for (auto &node : m->get_ordered_ops()) {
        if (ngraph::op::is_constant(node))
            continue;
        if (ngraph::op::is_parameter(node)) {
            SetNodeFusingType(node, NodeFusingType::IgnoredAfterInputs);
            continue;
        } else if (isSuitableConvolutionParent(node)) {
            // Initiate fusing chain
            SetNodeFusingType(node, NodeFusingType::FusedWithConvolution);
            continue;
        } else if (isSuitableBinaryConvolutionParent(node)) {
            SetNodeFusingType(node, NodeFusingType::FusedWithBinaryConvolution);
            continue;
        } else if (isSuitableMiscParent(node, channelAxis)) {
            SetNodeFusingType(node, NodeFusingType::FusedWithMisc);
            continue;
        } else if (isSuitableMatMulParent(node)) {
            SetNodeFusingType(node, NodeFusingType::FusedWithMatMul);
            continue;
        }
        for (const auto fusingChainType : getContinuableChains(node)) {
            if (isSuitableChildForFusingSimple(node, channelAxis)) {
                PropagateIfHasOnlyChild(node, fusingChainType);
            } else if (fusingChainType == NodeFusingType::FusedWithConvolution ||
                       fusingChainType == NodeFusingType::FusedWithBinaryConvolution) {
                if (isSuitableParentForFusingSumActivation(node)) {
                    PropagateIfHasOnlyChild(node, NodeFusingType::FusedWithConvolutionSumActivation);
                    // Mimic FuseConvolutionAndSimpleOperationThroughMaxPool
                } else if (isSuitablePoolChild(node)) {
                    PropagateIfHasOnlyChild(node, fusingChainType);
                }
            } else if (fusingChainType == NodeFusingType::FusedWithConvolutionSumActivation &&
                       isSuitableChildForFusingSumActivation(node)) {
                // Todo: Chain could be converted from FusedWithBinaryConvolution to FusedWithConvolution at this point
                // Set FusedWithConvolution, so the fusing chain could be propagated
                PropagateIfHasOnlyChild(node, NodeFusingType::FusedWithConvolution);
            } else if (fusingChainType == NodeFusingType::FusedWithMatMul) {
                // Handle fusings for both MatMul and FullyConnected
                NodeFusingType updatedChainType = fusingChainType;
                if (isSuitableChildForFusingMatMul(node, updatedChainType))
                    PropagateIfHasOnlyChild(node, updatedChainType);
            } else if (fusingChainType == NodeFusingType::IgnoredAfterInputs && (snippets::pass::AppropriateForSubgraph(node) ||
                        ov::is_type<ngraph::op::v0::Convert>(node) || ov::is_type<ngraph::op::v1::Transpose>(node))) {
                // In OV_API 2.0 after Input node with I8/U8 precisions incerts Convert node, moreother on TF models inserts
                // Transpose layer. These brakes an idea to leave Eltwise node with I8/U8 inputs and FP32 outputs instead of Subgrath node
                // TODO Remove an additional check on Convert/Transpose here after enabling Subgraths with I8/U8 inputs and FP32 outputs
                SetNodeFusingType(node, NodeFusingType::IgnoredAfterInputs);
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
