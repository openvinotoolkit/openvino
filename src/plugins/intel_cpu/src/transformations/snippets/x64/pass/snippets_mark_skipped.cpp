// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "snippets_mark_skipped.hpp"
#include "snippets/pass/tokenization.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/utils.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <utils/general_utils.h>
#include <utils/cpu_utils.hpp>

#include "itt.hpp"

using namespace ngraph;

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
            if (++numNonConstInputs != 1)
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
            if (!isPerTensorOrPerChannelBroadcastable(dataShape.get_max_shape(), weightShape.get_max_shape(), channelAxis, true))
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

inline bool canBeMatMulExecutedInInt8(const ov::element::Type& firstType, const ov::element::Type& secondType) {
    return one_of(firstType, ov::element::i8, ov::element::u8) && secondType == ov::element::i8;
}

bool SupportsFusingWithConvolution_Simple(const std::shared_ptr<const Node> &node, const int channelAxis = DEFAULT_AXIS) {
    return SupportsFusingWithConvolution_SumActivation(node) ||
           ov::is_type<ngraph::op::Tanh>(node) ||
           ov::is_type<ngraph::op::v0::Gelu>(node) ||
           ov::is_type<ngraph::op::v7::Gelu>(node) ||
           ov::is_type<ngraph::op::Abs>(node) ||
           ov::is_type<ngraph::op::Sqrt>(node) ||
           ov::is_type<ngraph::op::FakeQuantize>(node) ||
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
    int channelAxis = DEFAULT_AXIS;
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
bool isSuitableMiscParent(const std::shared_ptr<const Node> &node) {
    const bool is_suitable_node = ov::is_type<ngraph::op::v0::MVN>(node) ||
                                  ov::is_type<ngraph::op::v6::MVN>(node) ||
                                  ov::is_type<ngraph::op::v0::NormalizeL2>(node) ||
                                  ov::is_type<ngraph::op::v0::Interpolate>(node) ||
                                  ov::is_type<ngraph::op::v4::Interpolate>(node) ||
                                  ov::is_type<ngraph::op::v0::LSTMCell>(node) ||
                                  ov::is_type<ngraph::op::v4::LSTMCell>(node) ||
                                  ov::is_type<ngraph::opset1::ConvolutionBackpropData>(node) ||
                                  ov::is_type<ngraph::op::util::ArithmeticReductionKeepDims>(node) ||
                                  ov::is_type<ngraph::opset1::GroupConvolutionBackpropData>(node) ||
                                  ov::is_type<ngraph::opset1::AvgPool>(node);
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
// From Reduce::canFuse() corner case. CanFuseSimpleOperation is covered by Misc
inline bool isSuitableReduceParent(const std::shared_ptr<const Node> &node) {
    bool is_suitable_reduce = ov::is_type<ov::op::util::ArithmeticReductionKeepDims>(node) && isSuitableMiscParent(node);
    bool is_not_min_max = !ov::is_type<ov::op::v1::ReduceMax>(node) && !ov::is_type<ov::op::v1::ReduceMin>(node);
    bool out_is_f32 = node->get_output_element_type(0) == ov::element::f32;
    return is_suitable_reduce && is_not_min_max && out_is_f32;
}
// Subtract as ZeroPoints for Convolution
bool isSuitableSubtractAsZeroPointsParent(const std::shared_ptr<const Node> &node) {
    const bool is_suitable_node = ov::is_type<ngraph::op::v1::Subtract>(node);
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    const bool has_two_parents = node->get_input_size() == 2;
    if (!(is_suitable_node && has_only_child && has_two_parents))
        return false;

    const auto child = node->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
    const bool is_conv = ov::is_type<ov::op::v1::Convolution>(child);
    const bool is_group_conv = ov::is_type<ov::op::v1::GroupConvolution>(child);
    if (!is_conv && !is_group_conv)
        return false;
    const auto weight_pshape = child->get_input_partial_shape(1);
    if (weight_pshape.is_dynamic())
        return false;
    const auto weight_shape = weight_pshape.get_shape();
    const bool is_depthwise = is_group_conv && weight_shape[1] == 1 && weight_shape[2] == 1;
    const auto depthwise_rank = child->get_input_partial_shape(0).rank();
    if (depthwise_rank.is_dynamic())
        return false;
    const bool deptwise_is_suitable = implication(is_depthwise, depthwise_rank.get_length() < 5);
    if (!deptwise_is_suitable)
        return false;

    const auto zp_weights = node->get_input_node_shared_ptr(1);
    const auto zp_weight_pshape = zp_weights->get_output_partial_shape(0);
    if (zp_weight_pshape.is_dynamic())
        return false;
    const auto zp_weight_shape = zp_weight_pshape.get_shape();
    auto correct_shape = ov::Shape(zp_weight_shape.size(), 1);
    if (zp_weight_shape.size() > 1)
        correct_shape[1] = zp_weight_shape[1];
    const bool zp_weights_is_suitable = ov::is_type<ov::op::v0::Constant>(zp_weights) &&
                                        zp_weights->get_element_type() == ov::element::u8 &&
                                        zp_weight_shape.size() >= 2 && correct_shape == zp_weight_shape;
    const bool first_conv_input_is_suitable = node->get_input_element_type(0) == ov::element::u8 &&
                                              zp_weights_is_suitable;

    const auto conv_weights = child->get_input_node_shared_ptr(1);
    bool second_conv_input_is_suitable = ov::is_type<ngraph::op::v0::Constant>(conv_weights) &&
                                         conv_weights->get_output_element_type(0) == ov::element::i8;
    return first_conv_input_is_suitable && second_conv_input_is_suitable;
}
bool isSuitablePoolChild(const std::shared_ptr<const Node> &node) {
    const bool is_suitable_node = ov::is_type<ngraph::op::v1::MaxPool>(node);
    // has a single output, connected to a single child
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_suitable_node && has_only_child;
}
bool isSuitableChildForFusingSimple(const std::shared_ptr<const Node> &node, const int channelAxis = DEFAULT_AXIS) {
    // Note: Fusing child is allowed to have several users, but that must be the end of the chain
    return SupportsFusingWithConvolution_Simple(node, channelAxis) && getNumNonConstInputs(node) == 1;
}
bool isSuitableChildForFusingMatMul(const std::shared_ptr<const Node> &node, const bool canMatMulBeExecutedInI8,
                                    NodeFusingType &updatedChainType, int& fusingAxis) {
    int num_non_const_inputs = 0;
    bool can_be_converted_to_FC = false;
    ov::PartialShape bias_shape;
    ov::PartialShape matmul_shape;
    for (const auto &parent_out : node->input_values()) {
        const auto parent = parent_out.get_node_shared_ptr();
        if (ngraph::op::is_constant(parent)) {
            bias_shape = parent_out.get_shape();
            num_non_const_inputs++;
        } else {
              matmul_shape = parent_out.get_partial_shape();
              if (matmul_shape.size() == 0)
                return false;
            const auto& grandparents = parent->input_values();
            // first check that weights are constant and both activations and weights have static shape
            if (grandparents.size() == 2 &&
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

    // Matmul / FC bias fusion
    if (ov::is_type<ngraph::opset1::Add>(node) &&
        bias_shape.is_static() && matmul_shape.rbegin()->is_static() &&
        bias_shape.rbegin()->get_length() == matmul_shape.rbegin()->get_length() &&
        bias_shape.rbegin()->get_length() == shape_size(bias_shape.get_shape())) {
        return true;
    }

    // FuseMatMulAndSimpleOperation or FuseFullyConnectedAndSimpleOperation
    // Invoke SupportsFusingWithConvolution_Simple directly instead of isSuitableChildForFusingSimple to
    // eliminate getNumNonConstInputs() check
    fusingAxis = can_be_converted_to_FC ? (matmul_shape.size() == 3 ? 2 : 1) : matmul_shape.size() - 1;
    if (SupportsFusingWithConvolution_Simple(node, fusingAxis)) {
        updatedChainType = NodeFusingType::FusedWithMisc;
        return true;
    }

    // MatMul specific checks from ::canFuse()
    if (!can_be_converted_to_FC) {
        // can with rank() > 2
        // Algorithm::EltwisePowerStatic is ignored
        const auto rank = node->get_output_partial_shape(0).rank();
        if (rank.is_static() && rank.get_length() > 2) {
            if (ov::is_type<ov::op::v1::Add>(node) ||
                ov::is_type<ov::op::v1::Multiply>(node) ||
                ov::is_type<ov::op::v1::Subtract>(node) ||
                ov::is_type<ov::op::v1::Divide>(node) ||
                ov::is_type<ov::op::v0::PRelu>(node)) {
                const auto const1 = ov::is_type<ov::op::v0::Constant>(node->get_input_node_shared_ptr(0));
                const auto const2 = ov::is_type<ov::op::v0::Constant>(node->get_input_node_shared_ptr(1));
                int constPort = -1;
                if (const2) {
                    constPort = 1;
                } else if (const1) {
                    constPort = 0;
                }

                if (constPort != -1) {
                    auto const_shape = node->get_input_shape(constPort);
                    if (ov::shape_size(const_shape) != 1) {
                        return false;
                    }
                }
            } else if (ov::is_type<ov::op::v0::FakeQuantize>(node)) {
                const bool is_per_tensor_broadcasting = ngraph::snippets::utils::is_scalar_constant(node->get_input_node_shared_ptr(1)) &&
                                                        ngraph::snippets::utils::is_scalar_constant(node->get_input_node_shared_ptr(2)) &&
                                                        ngraph::snippets::utils::is_scalar_constant(node->get_input_node_shared_ptr(3)) &&
                                                        ngraph::snippets::utils::is_scalar_constant(node->get_input_node_shared_ptr(4));
                if (!is_per_tensor_broadcasting) {
                    return false;
                }
            }
        }

        // specific case for FQ
        if (ov::is_type<ov::op::v0::FakeQuantize>(node)) {
            if (one_of(node->get_output_element_type(0), ov::element::i8, ov::element::u8) && canMatMulBeExecutedInI8) {
                return false;
            }
        }
    }

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
    auto isFusedFQNode = [&isFusedBiasNode](std::shared_ptr<Node> n) {
        if (!(ov::is_type<ngraph::op::v0::FakeQuantize>(n) &&
            GetNodeFusingType(n) == NodeFusingType::FusedWithConvolution))
            return false;
        const auto& parent = n->get_input_node_shared_ptr(0);
        const bool is_suitable_parent = isSuitableConvolutionParent(parent)
            || isFusedBiasNode(parent)
            || (GetNodeFusingType(parent) == NodeFusingType::FusedWithConvolution);
        return is_suitable_parent;
    };
    int num_conv_parents = 0;
    for (size_t i = 0; i < node->get_input_size(); i++) {
        const auto n = node->get_input_node_shared_ptr(i);
        //BinaryConvolution allows other ops to be fused before the Add, while Convolution doesn't
        num_conv_parents += (isSuitableConvolutionParent(n) || isFusedBiasNode(n) || isFusedFQNode(n) ||
                             GetNodeFusingType(n) == NodeFusingType::FusedWithBinaryConvolution);
    }
    return getNumNonConstInputs(node) == 2 && num_conv_parents >=1;
}
bool isSuitableChildForFusingSumActivation(const std::shared_ptr<const Node> &node) {
    return SupportsFusingWithConvolution_SumActivation(node);
}
bool isSuitableReduceChild(const std::shared_ptr<const Node> &node, const int channelAxis = DEFAULT_AXIS) {
    return node->get_output_element_type(0) == ov::element::f32 && isSuitableChildForFusingSimple(node, channelAxis);
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
    if (!ov::is_type<ngraph::op::Convert>(node))
        return false;
    auto hasResult = [](const std::shared_ptr<const Node>& node){
        auto consumers = node->output(0).get_target_inputs();
        bool findResult = false;
        if (consumers.size() == 1) {
            if (ov::is_type<ngraph::op::Result>(consumers.begin()->get_node()))
                findResult = true;
        }
        return findResult;
    };
    // 1. check Parameter->Convert 2. check Convert->Result
    if (ov::is_type<ngraph::op::Parameter>(node->get_input_node_ptr(0))) {
        auto inPrc = node->get_input_element_type(0);
        auto outPrc = node->get_output_element_type(0);
        return inPrc == element::bf16 && outPrc == element::f32;
    } else if (hasResult(node)) {
        auto inPrc = node->get_input_element_type(0);
        auto outPrc = node->get_output_element_type(0);
        return inPrc == element::f32 && outPrc == element::bf16;
    } else {
        return false;
    }
}
} // namespace

bool SnippetsMarkSkipped::run_on_model(const std::shared_ptr<ov::Model> &m) {
    RUN_ON_MODEL_SCOPE(SnippetsMarkSkipped);
    int channelAxis = DEFAULT_AXIS;
    for (auto &node : m->get_ordered_ops()) {
        if (ngraph::op::is_constant(node) || ov::is_type<ov::op::v0::Result>(node))
            continue;
        if (isSuitableConvolutionParent(node)) {
            // Initiate fusing chain
            SetNodeFusingType(node, NodeFusingType::FusedWithConvolution);
            channelAxis = DEFAULT_AXIS;
        } else if (isSuitableBinaryConvolutionParent(node)) {
            SetNodeFusingType(node, NodeFusingType::FusedWithBinaryConvolution);
            channelAxis = DEFAULT_AXIS;
        } else if (isSuitableReduceParent(node)) {
            const auto reduce = std::dynamic_pointer_cast<const ngraph::op::util::ArithmeticReductionKeepDims>(node);
            channelAxis = getChannelAxis(reduce->get_reduction_axes(), reduce->get_keep_dims());
            SetNodeFusingType(node, NodeFusingType::FusedWithReduce);
        } else if (isSuitableMiscParent(node)) {
            if (const auto reduce = std::dynamic_pointer_cast<const ngraph::op::util::ArithmeticReductionKeepDims>(node)) {
                channelAxis = getChannelAxis(reduce->get_reduction_axes(), reduce->get_keep_dims());
            } else {
                channelAxis = DEFAULT_AXIS;
            }
            SetNodeFusingType(node, NodeFusingType::FusedWithMisc);
        } else if (isSuitableMatMulParent(node)) {
            if (canBeMatMulExecutedInInt8(node->get_input_element_type(0), node->get_input_element_type(1)))
                SetNodeFusingType(node, NodeFusingType::FusedWithMatMulI8);
            else
                SetNodeFusingType(node, NodeFusingType::FusedWithMatMul);
            channelAxis = DEFAULT_AXIS;
        } else if (isSuitableSubtractAsZeroPointsParent(node)) {
            SetSnippetsNodeType(node, snippets::pass::SnippetsNodeType::SkippedByPlugin);
            channelAxis = DEFAULT_AXIS;
        // CVS-105447
        // This WA skip convert with same I/O precision in Snippets
        // Such useless Convert is executed in Snippets
        } else if (enableBF16 && isSuitableConvert(node)) {
            SetSnippetsNodeType(node, snippets::pass::SnippetsNodeType::SkippedByPlugin);
            channelAxis = DEFAULT_AXIS;
        } else {
            for (const auto fusingChainType : getContinuableChains(node)) {
                if (fusingChainType == NodeFusingType::FusedWithReduce) {
                    if (isSuitableReduceChild(node, channelAxis))
                        PropagateIfHasOnlyChild(node, fusingChainType);
                } else if (isSuitableChildForFusingSimple(node, channelAxis)) {
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
                } else if (fusingChainType == NodeFusingType::FusedWithMatMul ||
                           fusingChainType == NodeFusingType::FusedWithMatMulI8) {
                    const bool isExecutedInINT8 = fusingChainType == NodeFusingType::FusedWithMatMulI8;
                    // Handle fusings for both MatMul and FullyConnected
                    NodeFusingType updatedChainType = fusingChainType;
                    if (isSuitableChildForFusingMatMul(node, isExecutedInINT8, updatedChainType, channelAxis))
                        PropagateIfHasOnlyChild(node, updatedChainType);
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
