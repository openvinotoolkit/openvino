// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "snippets/pass/filter_fused.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "snippets/register_info.hpp"
#include <ngraph/opsets/opset1.hpp>
#include "legacy/ngraph_ops/fully_connected.hpp"

namespace ngraph {
namespace snippets {
namespace pass {

namespace {
bool hasFusedParent(std::shared_ptr<Node> node, SnippetsNodeType& FusedChainType) {
    // todo: what if a node has > 1 parents with different fusing types?
    const std::vector<SnippetsNodeType> supportedFusingTypes = {SnippetsNodeType::FusedWithConvolution,
                                                                SnippetsNodeType::FusedWithConvolutionSumActivation,
                                                                SnippetsNodeType::FusedWithMisc};
    for (const auto& input : node->inputs()) {
        const auto parent = input.get_source_output().get_node_shared_ptr();
        for (auto s : supportedFusingTypes) {
            if (GetSnippetsNodeType(parent) == s) {
                FusedChainType = s;
                return true;
            }
        }
    }
    return false;
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
        if ((GetSnippetsNodeType(parent) == SnippetsNodeType::SubgraphStart) ||
            (GetSnippetsNodeType(parent) == SnippetsNodeType::SubgraphBody))
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
bool SupportsFusingWithConvolution_Simple(std::shared_ptr<Node> node) {
    //  This is an approximate solution. Do ann bynaries are supported?
    //   node->canBePerformedAsScaleShift(this);
    if (ngraph::op::is_binary_elementwise_arithmetic(node) ||
        SupportsFusingWithConvolution_SumActivation(node) ||
        ov::is_type<ngraph::op::Tanh>(node) ||
        ov::is_type<ngraph::op::Gelu>(node) ||
        ov::is_type<ngraph::op::Abs>(node) ||
        ov::is_type<ngraph::op::Sqrt>(node))
        return true;
    else
        return false;
}
// Convolution is a special case, since it supports peculiar fusings
bool isSuitableConvolutionParent(std::shared_ptr<Node> node) {
    const bool is_suitable_node = ov::is_type<ngraph::op::v1::Convolution>(node) ||
                                  ov::is_type<ngraph::op::v1::GroupConvolution>(node) ||
                                  ov::is_type<ngraph::op::v1::BinaryConvolution>(node);
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
                                  ov::is_type<ngraph::opset1::GroupConvolutionBackpropData>(node) ||
                                  ov::is_type<ngraph::op::MatMul>(node);
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
    if ( !SupportsFusingWithConvolution_Simple(node) )
        return false;
    return (getNumNonConstInputs(node) == 1);
}
bool isSuitableParentForFusingSumActivation(std::shared_ptr<Node> node) {
    if (dynamic_cast<const ngraph::op::v1::Add *>(node.get()) == nullptr)
        return false;
    int num_conv_parents = 0;
    for (size_t i = 0; i < node->get_input_size(); i++) {
        num_conv_parents += isSuitableConvolutionParent(node->get_input_node_shared_ptr(i));
    }
    const auto out = node->outputs();
    // Add always has one child, but the child should also have only one parent
    const bool has_only_child = (out[0].get_target_inputs().size() == 1);
    return getNumNonConstInputs(node) == 2 && has_only_child;
}
bool isSuitableChildForFusingSumActivation(std::shared_ptr<Node> node) {
    if ( !SupportsFusingWithConvolution_SumActivation(node) )
        return false;
    const auto out = node->outputs();
    // Activations always has one child, but the child should also have only one parent
    const bool has_only_child = (out[0].get_target_inputs().size() == 1);
    return has_only_child;
}
} // namespace

SnippetsNodeType GetSnippetsNodeType(std::shared_ptr<Node> node) {
    auto &rt = node->get_rt_info();
    const auto rinfo = rt.find("MayBeFusedInPlugin");
    if (rinfo == rt.end())
        return SnippetsNodeType::NotSet;
    const int64_t type_val = ov::as_type_ptr<ngraph::VariantWrapper<int64_t>>(rinfo->second)->get();
    const int64_t lower_bound = static_cast<int64_t>(SnippetsNodeType::FusedWithConvolution);
    const int64_t upper_bound = static_cast<int64_t>(SnippetsNodeType::SubgraphBody);
    if ((type_val < lower_bound) || (type_val > upper_bound))
        throw ngraph_error("Invalid value of SnippetsNodeType is detected.");
    return static_cast<SnippetsNodeType>(type_val);
}
void SetSnippetsNodeType(std::shared_ptr<Node> node, SnippetsNodeType nodeType) {
    auto &rt = node->get_rt_info();
    if (nodeType == SnippetsNodeType::NotSet) {
        throw ngraph_error("Attempt to set an invalid value of a SnippetsNodeType.");
    }
    rt["MayBeFusedInPlugin"] = std::make_shared<VariantWrapper<int64_t>>(VariantWrapper<int64_t>(static_cast<int64_t>(nodeType)));
}

bool FilterFused::run_on_function(std::shared_ptr<Function> f) {
    RUN_ON_FUNCTION_SCOPE(FulterFused);
    for (auto node : f->get_ordered_ops()) {
        if (ngraph::op::is_constant(node) || ngraph::op::is_parameter(node))
            continue;
        if (isSuitableConvolutionParent(node)) {
            // Initiate fusing chain
            SetSnippetsNodeType(node, SnippetsNodeType::FusedWithConvolution);
            continue;
        } else if (isSuitableMiscParent(node)) {
            SetSnippetsNodeType(node, SnippetsNodeType::FusedWithMisc);
            continue;
        }
        SnippetsNodeType fusingChainType{SnippetsNodeType::NotSet};
        if (hasFusedParent(node, fusingChainType)) {
            if (isSuitableChildForFusingSimple(node)) {
                SetSnippetsNodeType(node, fusingChainType);
            } else if (fusingChainType == SnippetsNodeType::FusedWithConvolution &&
                        isSuitableParentForFusingSumActivation(node)) {
                SetSnippetsNodeType(node, SnippetsNodeType::FusedWithConvolutionSumActivation);
            } else if (fusingChainType == SnippetsNodeType::FusedWithConvolutionSumActivation &&
                        isSuitableChildForFusingSumActivation(node)) {
                // Set FusedWithConvolution, so the fusing chain could be propagated
                SetSnippetsNodeType(node, SnippetsNodeType::FusedWithConvolution);
            // Mimic FuseConvolutionAndSimpleOperationThroughMaxPool
            } else if (fusingChainType == SnippetsNodeType::FusedWithConvolution && isSuitablePoolChild(node)) {
                SetSnippetsNodeType(node, SnippetsNodeType::FusedWithConvolution);
            }
        }
        if (AppropriateForSubgraph(node)) {
            // todo: enable u8 support in Snippetst
            // Ignore eltwise chains starting at Parameter node, since it could be u8
            if (hasIgnoredParent(node) || hasParameterParent(node)) {
                SetSnippetsNodeType(node, SnippetsNodeType::Ignored);
                continue;
            }
            if (GetSnippetsNodeType(node) == SnippetsNodeType::FusedWithMisc ||
                GetSnippetsNodeType(node) == SnippetsNodeType::FusedWithConvolution ||
                GetSnippetsNodeType(node) == SnippetsNodeType::FusedWithConvolutionSumActivation)
                continue;
            if (hasParentInStartedSubgraph (node))
                SetSnippetsNodeType(node, SnippetsNodeType::SubgraphBody);
            else
                SetSnippetsNodeType(node, SnippetsNodeType::SubgraphStart);
        }
    }
    return true;
}
} // namespace pass
} // namespace snippets
} // namespace ngraph
