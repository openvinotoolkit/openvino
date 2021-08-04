// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "snippets/pass/filter_fused.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "snippets/register_info.hpp"
#include <ngraph/opsets/opset1.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

namespace {
bool hasFusedParent(std::shared_ptr<Node> node) {
    for (const auto& input : node->inputs()) {
        const auto parent = input.get_source_output().get_node_shared_ptr();
        if (GetSnippetsNodeType(parent) == SnippetsNodeType::Fused)
            return true;
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
        if (!!as_type_ptr<ngraph::op::Parameter>(parent))
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
        if (!!as_type_ptr<ngraph::op::v1::Reshape>(parent)) {
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
    return  !!as_type_ptr<ngraph::op::Relu>(node) ||
            !!as_type_ptr<ngraph::op::PRelu>(node) ||
            !!as_type_ptr<ngraph::op::Elu>(node) ||
            !!as_type_ptr<ngraph::op::Sigmoid>(node) ||
            !!as_type_ptr<ngraph::op::v5::HSigmoid>(node) ||
            !!as_type_ptr<ngraph::op::Clamp>(node) ||
            !!as_type_ptr<ngraph::op::v4::Swish>(node) ||
            !!as_type_ptr<ngraph::op::v4::HSwish>(node) ||
            !!as_type_ptr<ngraph::op::v4::Mish>(node) ||
            !!as_type_ptr<ngraph::op::v5::Round>(node);
}
bool SupportsFusingWithConvolution_Simple(std::shared_ptr<Node> node) {
    //  This is an approximate solution. Do ann bynaries are supported?
    //   node->canBePerformedAsScaleShift(this);
    if (ngraph::op::is_binary_elementwise_arithmetic(node) ||
        SupportsFusingWithConvolution_SumActivation(node) ||
        !!as_type_ptr<ngraph::op::Tanh>(node) ||
        !!as_type_ptr<ngraph::op::Gelu>(node) ||
        !!as_type_ptr<ngraph::op::Abs>(node) ||
        !!as_type_ptr<ngraph::op::Sqrt>(node))
        return true;
    else
        return false;
}
bool isSutableParentForFusingSimple(std::shared_ptr<Node> node) {
    const bool is_suitable_node = (!!as_type_ptr<ngraph::op::v1::Convolution>(node) ||
                                 !!as_type_ptr<ngraph::op::v1::GroupConvolution>(node) ||
                                 !!as_type_ptr<ngraph::op::v1::BinaryConvolution>(node) ||
                                 !!as_type_ptr<ngraph::op::v0::MVN>(node) ||
                                 !!as_type_ptr<ngraph::op::v0::NormalizeL2>(node));
    // has a single output, connected to a single child
    const auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_suitable_node && has_only_child;
}

bool isSutableChildForFusingSimple(std::shared_ptr<Node> node) {
    if ( !SupportsFusingWithConvolution_Simple(node) )
        return false;
    return (getNumNonConstInputs(node) == 1);
}
bool isSutableChildForFusingSumActivation(std::shared_ptr<Node> node) {
    if (dynamic_cast<const ngraph::op::v1::Add *>(node.get()) == nullptr)
        return false;
    int num_non_const_inputs = 0;
    for (size_t i = 0; i < node->get_input_size(); i++) {
        if (!ngraph::op::is_constant(node->get_input_node_shared_ptr(i)))
            num_non_const_inputs++;
    }
    return (getNumNonConstInputs(node) == 2);
}
} // namespace

SnippetsNodeType GetSnippetsNodeType(std::shared_ptr<Node> node) {
    auto &rt = node->get_rt_info();
    const auto rinfo = rt.find("MayBeFusedInPlugin");
    if (rinfo == rt.end())
        return SnippetsNodeType::NotSet;
    const int64_t type_val = ngraph::as_type_ptr<ngraph::VariantWrapper<int64_t>>(rinfo->second)->get();
    const int64_t lower_bound = static_cast<int64_t>(SnippetsNodeType::Fused);
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
        if (isSutableParentForFusingSimple(node)) {
            // Initiate fusing chain
            SetSnippetsNodeType(node, SnippetsNodeType::Fused);
            continue;
        }
        if (hasFusedParent(node)) {
            if (isSutableChildForFusingSimple(node)) {
                // This feature is disabled to emulate FusingSimple->FusingActivationAndSum->FusingSimple
                // todo: clean all the commented code after benchmark and analysis
//                 // If the node is already marked, it was processed as a child activation in FusingSumActivation
//                if (!fused_tag_is_set(node))
                SetSnippetsNodeType(node, SnippetsNodeType::Fused);
            } else if (isSutableChildForFusingSumActivation(node)) {
                SetSnippetsNodeType(node, SnippetsNodeType::Fused);
                // node has only 1 output, because it is Add
                auto child_inputs = node->get_output_target_inputs(0);
                if (child_inputs.size() == 1) {
                    auto child_node = node->get_users()[0];
                    if (SupportsFusingWithConvolution_SumActivation(child_node))
                        SetSnippetsNodeType(child_node, SnippetsNodeType::Fused);
//                    else {
//                         // Tag with 0, so this node would not be processed by FusingSimple
//                        child_rt["MayBeFusedInPlugin"] = std::make_shared<VariantWrapper<int64_t>>(VariantWrapper<int64_t>(0));
//                    }
                }
            }
        }
        if (AppropriateForSubgraph(node)) {
            // todo: enable u8 support in Snippetst
            // Ignore eltwise chains starting at Parameter node, since it could be u8
            if (hasIgnoredParent(node) || hasParameterParent(node)) {
                SetSnippetsNodeType(node, SnippetsNodeType::Ignored);
                continue;
            }
            if (GetSnippetsNodeType(node) == SnippetsNodeType::Fused)
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
