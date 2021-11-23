// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/filter_fused.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "snippets/register_info.hpp"
#include <ngraph/opsets/opset1.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

namespace {
std::vector<SnippetsNodeType> getContinuableChains(std::shared_ptr<Node> node) {
    std::vector<SnippetsNodeType> result;
    for (const auto& input : node->inputs()) {
        const auto parent = input.get_source_output().get_node_shared_ptr();
        const auto snt = GetSnippetsNodeType(parent);
        if (snt > SnippetsNodeType::FusedTerminator) {
            result.push_back(snt);
        }
    }
    return result;
}
bool hasIgnoredParent(std::shared_ptr<Node> node) {
    for (const auto& input : node->inputs()) {
        const auto parent = input.get_source_output().get_node_shared_ptr();
        if (GetSnippetsNodeType(parent) == SnippetsNodeType::Ignored)
            return true;
    }
    return false;
}
bool hasParameterParent(std::shared_ptr<Node> node) {
    for (const auto& input : node->inputs()) {
        const auto parent = input.get_source_output().get_node_shared_ptr();
        if (ov::is_type<ngraph::op::Parameter>(parent))
            return true;
    }
    return false;
}
bool hasParentInStartedSubgraph(std::shared_ptr<Node> node) {
    auto inputs = node->inputs();
    for (const auto& input : inputs) {
        const auto parent = input.get_source_output().get_node_shared_ptr();
        // True for SubgraphStart and SubgraphBody by convention
        if (GetSnippetsNodeType(parent) < SnippetsNodeType::NotSet)
            return true;
    }
    return false;
}
int getNumNonConstInputs(std::shared_ptr<Node> node) {
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
bool SupportsFusingWithConvolution_SumActivation(std::shared_ptr<Node> node) {
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

bool canBePerformedAsScaleShift(std::shared_ptr<Node> node) {
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
        const auto isPerTensorOrPerChannelBroadcastable = [](const ov::Shape& dataShape, const ov::Shape& weightShape) {
            // per-tensor broabcastable
            if (std::all_of(weightShape.begin(), weightShape.end(), [](size_t v){return v == 1;}))
                return true;
            if (weightShape.size() > dataShape.size())
                return false;
            // Normalize weightShape
            std::vector<size_t> normalizedWeigthShape{weightShape};
            for (size_t j = 0; j < (dataShape.size() - weightShape.size()); j++) {
                normalizedWeigthShape.insert(normalizedWeigthShape.begin(), 1);
            }
            // per-channel broadcastable
            for (size_t j = 0; j < normalizedWeigthShape.size(); j++) {
                if ((j == 1 && normalizedWeigthShape[1] != dataShape[1]) || (j != 1 && normalizedWeigthShape[j] != 1))
                    return false;
            }
            return true;
        };

        for (size_t i = 0; i < node->get_input_size(); i++) {
            if (i == fusingPort)
                continue;
            const ov::PartialShape weightShape = node->get_input_partial_shape(i);
            if (weightShape.is_dynamic() ||
                !isPerTensorOrPerChannelBroadcastable(dataShape.get_shape(), weightShape.get_shape()))
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

bool SupportsFusingWithConvolution_Simple(std::shared_ptr<Node> node) {
    return SupportsFusingWithConvolution_SumActivation(node) ||
        ov::is_type<ngraph::op::Tanh>(node) ||
        ov::is_type<ngraph::op::v0::Gelu>(node) ||
        ov::is_type<ngraph::op::v7::Gelu>(node) ||
        ov::is_type<ngraph::op::Abs>(node) ||
        ov::is_type<ngraph::op::Sqrt>(node) ||
        canBePerformedAsScaleShift(node);
}
// Convolution is a special case, since it supports peculiar fusings
bool isSuitableConvolutionParent(std::shared_ptr<Node> node) {
    const bool is_suitable_node = ov::is_type<ngraph::op::v1::Convolution>(node) ||
                                  ov::is_type<ngraph::op::v1::GroupConvolution>(node);
    // has a single output, connected to a single child
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_suitable_node && has_only_child;
}
bool isSuitableBinaryConvolutionParent(std::shared_ptr<Node> node) {
    const bool is_suitable_node = ov::is_type<ngraph::op::v1::BinaryConvolution>(node);
    // has a single output, connected to a single child
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_suitable_node && has_only_child;
}
bool isSuitableMiscParent(std::shared_ptr<Node> node) {
    const bool is_suitable_node = ov::is_type<ngraph::op::v0::MVN>(node) ||
                                  ov::is_type<ngraph::op::v6::MVN>(node) ||
                                  ov::is_type<ngraph::op::v0::NormalizeL2>(node) ||
                                  ov::is_type<ngraph::op::v0::Interpolate>(node) ||
                                  ov::is_type<ngraph::op::v4::Interpolate>(node) ||
                                  ov::is_type<ngraph::op::v0::LSTMCell>(node) ||
                                  ov::is_type<ngraph::op::v4::LSTMCell>(node) ||
                                  ov::is_type<ngraph::opset1::ConvolutionBackpropData>(node) ||
                                  ov::is_type<ngraph::opset1::GroupConvolutionBackpropData>(node);
    // has a single output, connected to a single child
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_suitable_node && has_only_child;
}
// Matmul is a special case, since it supports simple + bias fusings
bool isSuitableMatMulParent(std::shared_ptr<Node> node) {
    const bool is_suitable_node = ov::is_type<ngraph::op::MatMul>(node);
    // has a single output, connected to a single child
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_suitable_node && has_only_child;
}
bool isSuitablePoolChild(std::shared_ptr<Node> node) {
    const bool is_suitable_node = ov::is_type<ngraph::op::v1::MaxPool>(node);
    // has a single output, connected to a single child
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_suitable_node && has_only_child;
}
bool isSuitableChildForFusingSimple(std::shared_ptr<Node> node) {
    // Note: Fusing child is allowed to have several users, but that must be the end of the chain
    return SupportsFusingWithConvolution_Simple(node) && getNumNonConstInputs(node) == 1;
}
bool isSuitableChildForFusingMatMul(std::shared_ptr<Node> node, SnippetsNodeType &updatedChainType) {
    if (!ov::is_type<ngraph::opset1::Add>(node))
        return false;
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
            if (getNumNonConstInputs(parent) == 1)
                can_be_converted_to_FC = true;
            const auto pshape = parent_out.get_partial_shape();
            if (pshape.is_dynamic())
                return false;
            matmul_shape = pshape.get_shape();
        }
    }
    if (num_non_const_inputs != 1)
        return false;

    // FuseMatMulAndSimpleOperation or FuseFullyConnectedAndSimpleOperation
    // Invoke SupportsFusingWithConvolution_Simple directly instead of isSuitableChildForFusingSimple to
    // eliminate getNumNonConstInputs() check
    if (SupportsFusingWithConvolution_Simple(node) &&
        (!can_be_converted_to_FC || matmul_shape.size() != 3)) {
            updatedChainType = SnippetsNodeType::FusedWithMisc;
            return true;
    }
    //    FullyConnectedBiasFusion
    if (!can_be_converted_to_FC ||
        bias_shape.back() != matmul_shape.back() ||
        bias_shape.back() != shape_size(bias_shape)) {
        return false;
    }
    // Fusing chain must be interrupted after the node, since reshape will be inserted
    if (bias_shape.size() >= 2)
        updatedChainType = SnippetsNodeType::FusedTerminator;
    return true;
}
bool isSuitableParentForFusingSumActivation(std::shared_ptr<Node> node) {
    if (!ov::is_type<ngraph::op::v1::Add>(node))
        return false;
    auto isFusedBiasNode = [](std::shared_ptr<Node> n){
        if (!(ov::is_type<ngraph::op::v1::Add>(n) &&
                GetSnippetsNodeType(n) ==  SnippetsNodeType::FusedWithConvolution))
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
                GetSnippetsNodeType(n) == SnippetsNodeType::FusedWithBinaryConvolution);
    }
    return getNumNonConstInputs(node) == 2 && num_conv_parents >=1;
}
bool isSuitableChildForFusingSumActivation(std::shared_ptr<Node> node) {
    return SupportsFusingWithConvolution_SumActivation(node);
}
// Continue fusing chain of the passed type if the node has one child
// Otherwise mark node as FusedTerminator (Fused, but fusing chain is interrupted)
void PropagateIfHasOnlyChild(std::shared_ptr<Node> node, SnippetsNodeType nodeType) {
    const auto out = node->outputs();
    const bool has_only_child = out.size() == 1 && out[0].get_target_inputs().size() == 1;
    SetSnippetsNodeType(node, has_only_child ? nodeType : SnippetsNodeType::FusedTerminator);
}

void SetTopologicalOrder(std::shared_ptr<Node> node, int64_t order) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::SetTopologicalOrder")
    auto &rt = node->get_rt_info();
    rt["TopologicalOrder"] = std::make_shared<VariantWrapper<int64_t>>(static_cast<int64_t>(order));
}
} // namespace

SnippetsNodeType GetSnippetsNodeType(std::shared_ptr<Node> node) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::GetSnippetsNodeType")
    auto &rt = node->get_rt_info();
    const auto rinfo = rt.find("MayBeFusedInPlugin");
    if (rinfo == rt.end())
        return SnippetsNodeType::NotSet;
    const int64_t type_val = ov::as_type_ptr<ngraph::VariantWrapper<int64_t>>(rinfo->second)->get();
    // Todo: Remove the check from DEBUG also as soon as the PR is merged
    #ifdef DEBUG
    const int64_t lower_bound = static_cast<int64_t>(SnippetsNodeType::SubgraphStart);
    const int64_t upper_bound = static_cast<int64_t>(SnippetsNodeType::FusedWithMisc);
    if ((type_val < lower_bound) || (type_val > upper_bound))
        throw ngraph_error("Invalid value of SnippetsNodeType is detected.");
    #endif
    return static_cast<SnippetsNodeType>(type_val);
}
void SetSnippetsNodeType(std::shared_ptr<Node> node, SnippetsNodeType nodeType) {
    auto &rt = node->get_rt_info();
    rt["MayBeFusedInPlugin"] = std::make_shared<VariantWrapper<int64_t>>(static_cast<int64_t>(nodeType));
}

bool FilterFused::run_on_function(std::shared_ptr<Function> f) {
    RUN_ON_FUNCTION_SCOPE(FulterFused);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::FilterFused")
    auto ordered_ops = f->get_ordered_ops();
    for (size_t order = 0; order < ordered_ops.size(); order++) {
        auto &node = ordered_ops[order];
        if (ngraph::op::is_constant(node) || ngraph::op::is_parameter(node))
            continue;
        // Todo: we don't really have to set order for every node, just for subgraph parents and children would be enough
        SetTopologicalOrder(node, order);
        if (isSuitableConvolutionParent(node)) {
            // Initiate fusing chain
            SetSnippetsNodeType(node, SnippetsNodeType::FusedWithConvolution);
            continue;
        } else if (isSuitableBinaryConvolutionParent(node)) {
            SetSnippetsNodeType(node, SnippetsNodeType::FusedWithBinaryConvolution);
            continue;
        } else if (isSuitableMiscParent(node)) {
            SetSnippetsNodeType(node, SnippetsNodeType::FusedWithMisc);
            continue;
        } else if (isSuitableMatMulParent(node)) {
            SetSnippetsNodeType(node, SnippetsNodeType::FusedWithMatMul);
            continue;
        }
        for (const auto fusingChainType : getContinuableChains(node)) {
            if (isSuitableChildForFusingSimple(node)) {
                PropagateIfHasOnlyChild(node, fusingChainType);
            } else if (fusingChainType == SnippetsNodeType::FusedWithConvolution ||
                    fusingChainType == SnippetsNodeType::FusedWithBinaryConvolution) {
                if (isSuitableParentForFusingSumActivation(node)) {
                    PropagateIfHasOnlyChild(node, SnippetsNodeType::FusedWithConvolutionSumActivation);
                // Mimic FuseConvolutionAndSimpleOperationThroughMaxPool
                } else if (isSuitablePoolChild(node)) {
                    PropagateIfHasOnlyChild(node, fusingChainType);
                }
            } else if (fusingChainType == SnippetsNodeType::FusedWithConvolutionSumActivation &&
                        isSuitableChildForFusingSumActivation(node)) {
                // Todo: Chain could be converted from FusedWithBinaryConvolution to FusedWithConvolution at this point
                // Set FusedWithConvolution, so the fusing chain could be propagated
                PropagateIfHasOnlyChild(node, SnippetsNodeType::FusedWithConvolution);
            } else if (fusingChainType == SnippetsNodeType::FusedWithMatMul) {
                // Handle fusings for both MatMul and FullyConnected
                SnippetsNodeType updatedChainType = fusingChainType;
                if (isSuitableChildForFusingMatMul(node, updatedChainType))
                    PropagateIfHasOnlyChild(node, updatedChainType);
            }
        }
        if (AppropriateForSubgraph(node)) {
            // todo: enable u8 support in Snippetst
            // Ignore eltwise chains starting at Parameter node, since it could be u8
            if (hasIgnoredParent(node) || hasParameterParent(node)) {
                SetSnippetsNodeType(node, SnippetsNodeType::Ignored);
                continue;
            }
            if (GetSnippetsNodeType(node) >= SnippetsNodeType::FusedTerminator)
                continue;
            if (hasParentInStartedSubgraph(node)) {
                SetSnippetsNodeType(node, SnippetsNodeType::SubgraphBody);
            } else {
                SetSnippetsNodeType(node, SnippetsNodeType::SubgraphStart);
            }
        }
    }
    return true;
}
} // namespace pass
} // namespace snippets
} // namespace ngraph
