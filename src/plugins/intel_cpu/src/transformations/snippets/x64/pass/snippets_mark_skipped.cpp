// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "snippets_mark_skipped.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "itt.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/tokenization.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/utils.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

namespace {
static const int DEFAULT_AXIS = 1;
NodeFusingType GetNodeFusingType(const std::shared_ptr<const Node>& node) {
    auto& rt = node->get_rt_info();
    const auto rinfo = rt.find("MayBeFusedInPlugin");
    if (rinfo == rt.end()) {
        return NodeFusingType::NotSet;
    }
    return rinfo->second.as<NodeFusingType>();
}
void SetNodeFusingType(const std::shared_ptr<Node>& node, NodeFusingType nodeType) {
    auto& rt = node->get_rt_info();
    rt["MayBeFusedInPlugin"] = nodeType;
}
std::vector<NodeFusingType> getContinuableChains(const std::shared_ptr<const Node>& node) {
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
int getNumNonConstInputs(const std::shared_ptr<const Node>& node) {
    int num_non_const_inputs = 0;
    for (const auto& parent_out : node->input_values()) {
        const auto parent = parent_out.get_node_shared_ptr();
        if (ov::is_type<ov::op::v1::Reshape>(parent)) {
            for (const auto& grandparent_out : parent->input_values()) {
                const auto grandparent = grandparent_out.get_node_shared_ptr();
                if (!ov::is_type<ov::op::v0::Constant>(grandparent)) {
                    num_non_const_inputs++;
                }
            }
        } else if (!ov::is_type<ov::op::v0::Constant>(parent)) {
            num_non_const_inputs++;
        }
    }
    return num_non_const_inputs;
}
bool isFullyConnected(const std::shared_ptr<const ov::Node>& node) {
    if (!ov::is_type<ov::op::v0::MatMul>(node)) {
        return false;
    }
    const auto out_activations = node->input_value(0);
    const auto out_weights = node->input_value(1);
    const auto rank_a = out_activations.get_partial_shape().rank();
    const auto rank_w = out_weights.get_partial_shape().rank();
    return out_weights.get_partial_shape().is_static() && rank_a.is_static() && rank_w.is_static() &&
           rank_a.get_length() != 1 && rank_w.get_length() != 1 && rank_a.get_length() <= 3 &&
           rank_w.get_length() <= 3 && ov::op::util::is_on_constant_path(out_weights);
}
bool SupportsFusingWithConvolution_SumActivation(const std::shared_ptr<const Node>& node) {
    // todo: Do all PReLUs are fused? Not sure about round and softRelu
    // EltwiseRoundHalfToEven, EltwiseRoundHalfAwayFromZero, EltwiseSoftRelu
    return ov::is_type_any_of<ov::op::v0::Relu,
                              ov::op::v0::PRelu,
                              ov::op::v0::Elu,
                              ov::op::v0::Sigmoid,
                              ov::op::v5::HSigmoid,
                              ov::op::v0::Clamp,
                              ov::op::v4::Swish,
                              ov::op::v4::HSwish,
                              ov::op::v4::Mish,
                              ov::op::v5::Round>(node);
}

bool canBePerformedAsScaleShift(const std::shared_ptr<const Node>& node, const int channelAxis) {
    size_t fusingPort = 0;
    size_t numNonConstInputs = 0;
    ov::PartialShape dataShape;
    for (size_t i = 0; i < node->get_input_size(); i++) {
        const auto parent = node->get_input_node_shared_ptr(i);
        if (!ov::is_type<ov::op::v0::Constant>(parent)) {
            fusingPort = i;
            dataShape = node->get_input_partial_shape(i);
            // only one non-const parent is allowed
            if (++numNonConstInputs != 1) {
                return false;
            }
        } else {
            // every const parent must have exactly one child
            const auto out = parent->outputs();
            const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
            if (!has_only_child) {
                return false;
            }
        }
    }

    const auto isBroadcastableToDataInput = [&]() {
        for (size_t i = 0; i < node->get_input_size(); i++) {
            if (i == fusingPort) {
                continue;
            }
            const ov::PartialShape weightShape = node->get_input_partial_shape(i);
            if (!isPerTensorOrPerChannelBroadcastable(dataShape.get_max_shape(),
                                                      weightShape.get_max_shape(),
                                                      channelAxis,
                                                      true)) {
                return false;
            }
        }
        return true;
    };

    // Prelu and MulAdd are still ignored
    // isConvertablePowerStatic() is ignored
    return ov::is_type_any_of<ov::opset1::Add, ov::opset1::Multiply, ov::opset1::Subtract, ov::opset1::Divide>(node) &&
           isBroadcastableToDataInput();
}

inline bool canBeMatMulExecutedInInt8(const ov::element::Type& firstType, const ov::element::Type& secondType) {
    return one_of(firstType, ov::element::i8, ov::element::u8) && secondType == ov::element::i8;
}

bool SupportsFusingWithConvolution_Simple(const std::shared_ptr<const Node>& node,
                                          const int channelAxis = DEFAULT_AXIS) {
    return SupportsFusingWithConvolution_SumActivation(node) ||
           ov::is_type_any_of<ov::op::v0::Tanh,
                              ov::op::v0::Gelu,
                              ov::op::v7::Gelu,
                              ov::op::v0::Abs,
                              ov::op::v0::Sqrt,
                              ov::op::v0::FakeQuantize>(node) ||
           canBePerformedAsScaleShift(node, channelAxis);
}
// Convolution is a special case, since it supports peculiar fusings
bool isSuitableConvolutionParent(const std::shared_ptr<const Node>& node) {
    const bool is_suitable_node = ov::is_type_any_of<ov::op::v1::Convolution, ov::op::v1::GroupConvolution>(node);
    // has a single output, connected to a single child
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_suitable_node && has_only_child;
}
bool isSuitableBinaryConvolutionParent(const std::shared_ptr<const Node>& node) {
    const bool is_suitable_node = ov::is_type<ov::op::v1::BinaryConvolution>(node);
    // has a single output, connected to a single child
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_suitable_node && has_only_child;
}
int getChannelAxis(const ov::AxisSet& axes, bool keep_dims) {
    int channelAxis = DEFAULT_AXIS;
    if (!keep_dims) {
        for (auto& axis : axes) {
            if (axis == 1) {
                // channel axis has been reduced and doesn't exist any more
                channelAxis = -1;
                break;
            }
            if (axis == 0) {
                channelAxis = 0;
            }
        }
    }
    return channelAxis;
}
bool isSuitableMiscParent(const std::shared_ptr<const Node>& node) {
    const bool is_suitable_node = ov::is_type_any_of<ov::op::v0::MVN,
                                                     ov::op::v6::MVN,
                                                     ov::op::v0::NormalizeL2,
                                                     ov::op::v0::Interpolate,
                                                     ov::op::v4::Interpolate,
                                                     ov::op::v0::LSTMCell,
                                                     ov::op::v4::LSTMCell,
                                                     ov::opset1::ConvolutionBackpropData,
                                                     ov::op::util::ArithmeticReductionKeepDims,
                                                     ov::opset1::GroupConvolutionBackpropData,
                                                     ov::opset1::AvgPool,
                                                     ov::op::v14::AvgPool>(node);
    // has a single output, connected to a single child
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_suitable_node && has_only_child;
}
// Matmul is a special case, since it supports simple + bias fusings
bool isSuitableMatMulParent(const std::shared_ptr<const Node>& node) {
    const bool is_suitable_node = ov::is_type<ov::op::v0::MatMul>(node);
    // has a single output, connected to a single child
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_suitable_node && has_only_child;
}
// From Reduce::canFuse() corner case. CanFuseSimpleOperation is covered by Misc
inline bool isSuitableReduceParent(const std::shared_ptr<const Node>& node) {
    return ov::is_type<ov::op::util::ArithmeticReductionKeepDims>(node) && isSuitableMiscParent(node);
}
// Subtract as ZeroPoints for Convolution
bool isSuitableSubtractAsZeroPointsParent(const std::shared_ptr<const Node>& node) {
    const bool is_suitable_node = ov::is_type<ov::op::v1::Subtract>(node);
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    const bool has_two_parents = node->get_input_size() == 2;
    if (!(is_suitable_node && has_only_child && has_two_parents)) {
        return false;
    }

    const auto child = node->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
    const bool is_conv = ov::is_type<ov::op::v1::Convolution>(child);
    const bool is_group_conv = ov::is_type<ov::op::v1::GroupConvolution>(child);
    if (!is_conv && !is_group_conv) {
        return false;
    }
    const auto weight_pshape = child->get_input_partial_shape(1);
    if (weight_pshape.is_dynamic()) {
        return false;
    }
    const auto weight_shape = weight_pshape.get_shape();
    const bool is_depthwise = is_group_conv && weight_shape[1] == 1 && weight_shape[2] == 1;
    const auto depthwise_rank = child->get_input_partial_shape(0).rank();
    if (depthwise_rank.is_dynamic()) {
        return false;
    }
    const bool deptwise_is_suitable = implication(is_depthwise, depthwise_rank.get_length() < 5);
    if (!deptwise_is_suitable) {
        return false;
    }

    const auto zp_weights = node->get_input_node_shared_ptr(1);
    const auto zp_weight_pshape = zp_weights->get_output_partial_shape(0);
    if (zp_weight_pshape.is_dynamic()) {
        return false;
    }
    const auto zp_weight_shape = zp_weight_pshape.get_shape();
    auto correct_shape = ov::Shape(zp_weight_shape.size(), 1);
    if (zp_weight_shape.size() > 1) {
        correct_shape[1] = zp_weight_shape[1];
    }
    const bool zp_weights_is_suitable = ov::is_type<ov::op::v0::Constant>(zp_weights) &&
                                        zp_weights->get_element_type() == ov::element::u8 &&
                                        zp_weight_shape.size() >= 2 && correct_shape == zp_weight_shape;
    const bool first_conv_input_is_suitable =
        node->get_input_element_type(0) == ov::element::u8 && zp_weights_is_suitable;

    const auto conv_weights = child->get_input_node_shared_ptr(1);
    bool second_conv_input_is_suitable =
        ov::is_type<ov::op::v0::Constant>(conv_weights) && conv_weights->get_output_element_type(0) == ov::element::i8;
    return first_conv_input_is_suitable && second_conv_input_is_suitable;
}
bool isSuitablePoolChild(const std::shared_ptr<const Node>& node) {
    const bool is_suitable_node = ov::is_type<ov::op::v1::MaxPool>(node);
    // has a single output, connected to a single child
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_suitable_node && has_only_child;
}
bool isSuitableChildForFusingSimple(const std::shared_ptr<const Node>& node, const int channelAxis = DEFAULT_AXIS) {
    // Note: Fusing child is allowed to have several users, but that must be the end of the chain
    return SupportsFusingWithConvolution_Simple(node, channelAxis) && getNumNonConstInputs(node) == 1;
}

bool isSuitableChildForFusingMatMul(const std::shared_ptr<const Node>& node,
                                    const bool canMatMulBeExecutedInI8,
                                    NodeFusingType& updatedChainType,
                                    int& fusingAxis) {
    // Firsly check for Bias and DQScales fusion
    const bool is_bias = ov::is_type<ov::opset1::Add>(node);
    const bool is_dq_scales = ov::is_type<ov::opset1::Multiply>(node) && canMatMulBeExecutedInI8;
    if (is_bias || is_dq_scales) {
        for (const auto& in : node->inputs()) {
            const auto& parent_out = in.get_source_output();
            const auto& parent = parent_out.get_node_shared_ptr();
            const auto& parent_pshape = parent_out.get_partial_shape();
            if (ov::is_type<ov::op::v0::MatMul>(parent) && parent_pshape.rank().is_static()) {
                if (parent->get_output_target_inputs(0).size() > 1) {
                    break;
                }
                const auto bias_port = 1 - in.get_index();
                const auto bias_out = node->input_value(bias_port);
                if ((bias_out.get_target_inputs().size() > 1) || !ov::op::util::is_on_constant_path(bias_out)) {
                    break;
                }
                const auto& bias_pshape = bias_out.get_partial_shape();
                if (bias_pshape.is_dynamic()) {
                    break;
                }
                auto getNormalizedPShape = [](const ov::PartialShape& dims, size_t ndims) -> ov::PartialShape {
                    if (dims.size() >= ndims) {
                        return dims;
                    }
                    ov::PartialShape pshape(std::vector<size_t>(ndims, 1));
                    std::copy(dims.rbegin(), dims.rend(), pshape.rbegin());
                    return pshape;
                };
                const auto bias_pshape_norm = getNormalizedPShape(bias_pshape, parent_pshape.size());
                if (fusingAxis >= static_cast<int>(bias_pshape_norm.size()) ||
                    fusingAxis >= static_cast<int>(parent_pshape.size()) ||
                    bias_pshape_norm.size() != parent_pshape.size() || bias_pshape_norm.size() < 2) {
                    break;
                }
                if (((bias_pshape_norm[fusingAxis] == parent_pshape[fusingAxis]) ||
                     (is_dq_scales && bias_pshape_norm[fusingAxis] == 1)) &&
                    (bias_pshape_norm[fusingAxis] == static_cast<int64_t>(shape_size(bias_pshape_norm.get_shape())))) {
                    return true;
                }
            }
        }
    }

    // MatMul specific checks from ::canFuse()
    if (one_of(updatedChainType, NodeFusingType::FusedWithMatMul, NodeFusingType::FusedWithMatMulI8)) {
        const auto is_binary_eltwise = ov::is_type_any_of<ov::op::v1::Add,
                                                          ov::op::v1::Multiply,
                                                          ov::op::v1::Subtract,
                                                          ov::op::v1::Divide,
                                                          ov::op::v0::PRelu>(node);
        const auto rank = node->get_output_partial_shape(0).rank();
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) && rank.is_static() && is_binary_eltwise) {
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
                if (ov::shape_size(const_shape) != 1 && rank.get_length() > 4) {
                    return false;
                }
            }
        }

        if (ov::is_type<ov::op::v0::FakeQuantize>(node)) {
            if (one_of(node->get_output_element_type(0), ov::element::i8, ov::element::u8) &&
                !canMatMulBeExecutedInI8) {
                return false;
            }
        }
    }

    // FuseMatMulAndSimpleOperation or FuseFullyConnectedAndSimpleOperation
    // Invoke SupportsFusingWithConvolution_Simple directly instead of isSuitableChildForFusingSimple to
    // eliminate getNumNonConstInputs() check
    if (SupportsFusingWithConvolution_Simple(node, fusingAxis)) {
        size_t num_non_const_inputs = 0;
        size_t num_mm_inputs = 0;
        for (const auto& parent_out : node->input_values()) {
            // To avoid endless check `is_on_constant_path` for MatMul branch
            if (one_of(GetNodeFusingType(parent_out.get_node_shared_ptr()),
                       NodeFusingType::FusedWithMatMul,
                       NodeFusingType::FusedWithMatMulI8,
                       NodeFusingType::FusedWithFC,
                       NodeFusingType::FusedWithFCI8)) {
                num_mm_inputs++;
            } else if (!ov::op::util::is_on_constant_path(parent_out)) {
                num_non_const_inputs++;
            }
        }
        if (num_non_const_inputs + num_mm_inputs != 1) {
            return false;
        }

        updatedChainType = NodeFusingType::FusedWithMisc;
        return true;
    }

    return false;
}
bool isSuitableParentForFusingSumActivation(const std::shared_ptr<const Node>& node) {
    if (!ov::is_type<ov::op::v1::Add>(node)) {
        return false;
    }
    auto isFusedBiasNode = [](const std::shared_ptr<Node>& n) {
        if (!(ov::is_type<ov::op::v1::Add>(n) && GetNodeFusingType(n) == NodeFusingType::FusedWithConvolution)) {
            return false;
        }
        const auto conv = n->get_input_source_output(0);
        const auto bias = n->get_input_source_output(1);
        if (!(ov::is_type<ov::op::v0::Constant>(bias.get_node_shared_ptr()) &&
              isSuitableConvolutionParent(conv.get_node_shared_ptr()))) {
            return false;
        }
        const auto& conv_shape = conv.get_partial_shape();
        const auto& bias_shape = bias.get_partial_shape();
        if (conv_shape.rank().is_dynamic()) {
            return false;
        }
        auto getNormalizedDims = [](const ov::Shape& dims, size_t ndims) -> ov::Shape {
            ov::Shape normalizedDims = dims;
            for (size_t i = 0; i < (ndims - dims.size()); i++) {
                normalizedDims.insert(normalizedDims.begin(), 1);
            }
            return normalizedDims;
        };
        const auto bias_norm_dims = getNormalizedDims(bias_shape.get_shape(), conv_shape.size());
        if (conv_shape.size() != bias_norm_dims.size() || bias_norm_dims.size() < 2) {
            return false;
        }
        const auto channelAxis = 1;
        return conv_shape[channelAxis].is_static() &&
               conv_shape[channelAxis].get_length() == static_cast<int64_t>(bias_norm_dims[channelAxis]) &&
               bias_norm_dims[channelAxis] == static_cast<size_t>(shape_size(bias_norm_dims));
    };
    auto isFusedFQNode = [&isFusedBiasNode](const std::shared_ptr<Node>& n) {
        if (!(ov::is_type<ov::op::v0::FakeQuantize>(n) &&
              GetNodeFusingType(n) == NodeFusingType::FusedWithConvolution)) {
            return false;
        }
        const auto& parent = n->get_input_node_shared_ptr(0);
        const bool is_suitable_parent = isSuitableConvolutionParent(parent) || isFusedBiasNode(parent) ||
                                        (GetNodeFusingType(parent) == NodeFusingType::FusedWithConvolution);
        return is_suitable_parent;
    };
    int num_conv_parents = 0;
    for (size_t i = 0; i < node->get_input_size(); i++) {
        const auto n = node->get_input_node_shared_ptr(i);
        // BinaryConvolution allows other ops to be fused before the Add, while Convolution doesn't
        num_conv_parents += (isSuitableConvolutionParent(n) || isFusedBiasNode(n) || isFusedFQNode(n) ||
                             GetNodeFusingType(n) == NodeFusingType::FusedWithBinaryConvolution);
    }
    return getNumNonConstInputs(node) == 2 && num_conv_parents >= 1;
}
bool isSuitableChildForFusingSumActivation(const std::shared_ptr<const Node>& node) {
    return SupportsFusingWithConvolution_SumActivation(node);
}
bool isSuitableReduceChild(const std::shared_ptr<const Node>& node, const int channelAxis = DEFAULT_AXIS) {
    return isSuitableChildForFusingSimple(node, channelAxis);
}
bool isSuitableMatMulWithConstantPath(const std::shared_ptr<Node>& node) {
    return ov::is_type<ov::opset1::MatMul>(node) &&
           !ov::is_type<ov::opset1::Constant>(node->get_input_node_shared_ptr(1)) &&
           ov::op::util::is_on_constant_path(node->input_value(1));
}
// Continue fusing chain of the passed type if the node has one child
// Otherwise mark node as FusedTerminator (Fused, but fusing chain is interrupted)
void PropagateIfHasOnlyChild(const std::shared_ptr<Node>& node, NodeFusingType nodeType) {
    const auto out = node->outputs();
    const bool has_only_child = out.size() == 1 && out[0].get_target_inputs().size() == 1;
    SetNodeFusingType(node, has_only_child ? nodeType : NodeFusingType::FusedTerminator);
}
// todo: Skipping MultiSubGraphOp such as TensorIterator, Loop and If. Snippets might tokenize their bodies in the
// future.
//  Note that the function is recurrent, since there might be multi-level MultiSubGraphOp, if(){if(){}}else{} for
//  example.
void MarkSubgraphOpAsSkipped(const std::shared_ptr<Node>& node) {
    if (ov::is_type<ov::op::util::MultiSubGraphOp>(node)) {
        std::vector<std::shared_ptr<ov::Model>> models{};
        // Covers TensorIterator and Loop
        if (auto s = ov::as_type_ptr<ov::op::util::SubGraphOp>(node)) {
            models.push_back(s->get_function());
            // Add new multi-body subgraph op here
        } else if (auto if_op = ov::as_type_ptr<ov::op::v8::If>(node)) {
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
    if (!ov::is_type<ov::op::v0::Convert>(node)) {
        return false;
    }
    auto hasResult = [](const std::shared_ptr<const Node>& node) {
        auto consumers = node->output(0).get_target_inputs();
        bool findResult = false;
        if (consumers.size() == 1) {
            if (ov::is_type<ov::op::v0::Result>(consumers.begin()->get_node())) {
                findResult = true;
            }
        }
        return findResult;
    };
    // 1. check Parameter->Convert 2. check Convert->Result
    if (ov::is_type<ov::op::v0::Parameter>(node->get_input_node_ptr(0))) {
        auto inPrc = node->get_input_element_type(0);
        auto outPrc = node->get_output_element_type(0);
        return inPrc == element::bf16 && outPrc == element::f32;
    }
    if (hasResult(node)) {
        auto inPrc = node->get_input_element_type(0);
        auto outPrc = node->get_output_element_type(0);
        return inPrc == element::f32 && outPrc == element::bf16;
    }
    return false;
}

auto is_skipped_op(const std::shared_ptr<ov::Node>& op) -> bool {
    return ov::is_type_any_of<ov::op::v0::Constant, ov::op::v0::Parameter, ov::op::v0::Result>(op);
}
}  // namespace

bool SnippetsMarkSkipped::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(SnippetsMarkSkipped);
    int channelAxis = DEFAULT_AXIS;
    for (auto& node : m->get_ordered_ops()) {
        if (is_skipped_op(node)) {
            continue;
        }
        // We perform this check separately because we mark here only weights path
        // Matmul itself will be checked further
        if (isSuitableMatMulWithConstantPath(node)) {
            auto markup_func = [](Node* node) {
                SetSnippetsNodeType(node->shared_from_this(), snippets::pass::SnippetsNodeType::SkippedByPlugin);
            };
            std::unordered_set<Node*> visited;
            ov::op::util::visit_constant_path(node->get_input_node_ptr(1), visited, markup_func);
        }
        if (isSuitableConvolutionParent(node)) {
            // Initiate fusing chain
            SetNodeFusingType(node, NodeFusingType::FusedWithConvolution);
            channelAxis = DEFAULT_AXIS;
        } else if (isSuitableBinaryConvolutionParent(node)) {
            SetNodeFusingType(node, NodeFusingType::FusedWithBinaryConvolution);
            channelAxis = DEFAULT_AXIS;
        } else if (isSuitableReduceParent(node)) {
            const auto reduce = ov::as_type_ptr<const ov::op::util::ArithmeticReductionKeepDims>(node);
            channelAxis = getChannelAxis(reduce->get_reduction_axes(), reduce->get_keep_dims());
            SetNodeFusingType(node, NodeFusingType::FusedWithReduce);
        } else if (isSuitableMiscParent(node)) {
            if (const auto reduce = ov::as_type_ptr<const ov::op::util::ArithmeticReductionKeepDims>(node)) {
                channelAxis = getChannelAxis(reduce->get_reduction_axes(), reduce->get_keep_dims());
            } else {
                channelAxis = DEFAULT_AXIS;
            }
            SetNodeFusingType(node, NodeFusingType::FusedWithMisc);
        } else if (isSuitableMatMulParent(node)) {
            const bool is_fc = isFullyConnected(node);
            const bool is_i8 =
                canBeMatMulExecutedInInt8(node->get_input_element_type(0), node->get_input_element_type(1));
            const auto out_rank = node->get_output_partial_shape(0).rank();
            if (is_fc) {
                SetNodeFusingType(node, is_i8 ? NodeFusingType::FusedWithFCI8 : NodeFusingType::FusedWithFC);
                channelAxis = out_rank.is_static() ? (out_rank.get_length() == 3 ? 2 : 1) : DEFAULT_AXIS;
            } else {
                SetNodeFusingType(node, is_i8 ? NodeFusingType::FusedWithMatMulI8 : NodeFusingType::FusedWithMatMul);
                channelAxis = out_rank.is_static() ? out_rank.get_length() - 1 : DEFAULT_AXIS;
            }
        } else if (isSuitableSubtractAsZeroPointsParent(node) || (enableBF16 && isSuitableConvert(node))) {
            // CVS-105447
            // This WA skip convert with same I/O precision in Snippets
            // Such useless Convert is executed in Snippets
            SetSnippetsNodeType(node, snippets::pass::SnippetsNodeType::SkippedByPlugin);
            channelAxis = DEFAULT_AXIS;
        } else {
            for (const auto fusingChainType : getContinuableChains(node)) {
                if (fusingChainType == NodeFusingType::FusedWithReduce) {
                    if (isSuitableReduceChild(node, channelAxis)) {
                        PropagateIfHasOnlyChild(node, fusingChainType);
                    }
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
                    // Todo: Chain could be converted from FusedWithBinaryConvolution to FusedWithConvolution at this
                    // point Set FusedWithConvolution, so the fusing chain could be propagated
                    PropagateIfHasOnlyChild(node, NodeFusingType::FusedWithConvolution);
                } else if (one_of(fusingChainType,
                                  NodeFusingType::FusedWithMatMul,
                                  NodeFusingType::FusedWithMatMulI8,
                                  NodeFusingType::FusedWithFC,
                                  NodeFusingType::FusedWithFCI8)) {
                    const bool isExecutedInINT8 =
                        one_of(fusingChainType, NodeFusingType::FusedWithMatMulI8, NodeFusingType::FusedWithFCI8);
                    // Handle fusings for both MatMul and FullyConnected
                    NodeFusingType updatedChainType = fusingChainType;
                    if (isSuitableChildForFusingMatMul(node, isExecutedInINT8, updatedChainType, channelAxis)) {
                        PropagateIfHasOnlyChild(node, updatedChainType);
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

}  // namespace ov::intel_cpu
