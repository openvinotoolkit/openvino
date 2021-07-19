// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "snippets/pass/filter_fused.hpp"
#include "snippets/register_info.hpp"
#include <ngraph/opsets/opset1.hpp>

namespace {
using namespace ngraph;
//bool fused_tag_is_set(std::shared_ptr<Node> node) {
//    auto &rt = node->get_rt_info();
//    const auto rinfo = rt.find("MayBeFusedInPlugin");
//    return rinfo != rt.end();
//}
bool is_fused(std::shared_ptr<Node> node) {
    auto &rt = node->get_rt_info();
    const auto rinfo = rt.find("MayBeFusedInPlugin");
    if (rinfo == rt.end()) {
        return false;
    }
    int64_t may_be_fused = ngraph::as_type_ptr<ngraph::VariantWrapper<int64_t>>(rinfo->second)->get();
    return  (may_be_fused == 1);
}

bool has_fused_parent(std::shared_ptr<Node> node) {
    for (size_t i = 0; i < node->get_input_size(); i++) {
        auto parent = node->get_input_node_shared_ptr(i);
        if (is_fused(parent))
            return true;
    }
   return false;
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
    const bool is_convolution = (!!as_type_ptr<ngraph::op::v1::Convolution>(node) ||
                                 !!as_type_ptr<ngraph::op::v1::GroupConvolution>(node) ||
                                 !!as_type_ptr<ngraph::op::v1::BinaryConvolution>(node));
    // has a single output, connected to a single child
    auto out = node->outputs();
    const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
    return is_convolution && has_only_child;
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
bool ngraph::snippets::pass::FilterFused::run_on_function(std::shared_ptr<Function> f) {
    RUN_ON_FUNCTION_SCOPE(FulterFused);
    for (auto node : f->get_ordered_ops()) {
        if (ngraph::op::is_constant(node) || ngraph::op::is_parameter(node))
            continue;
        auto &rt = node->get_rt_info();
        if (isSutableParentForFusingSimple(node)) {
            // Initiate fusing chain
            rt["MayBeFusedInPlugin"] = std::make_shared<VariantWrapper<int64_t>>(VariantWrapper<int64_t>(1));
            continue;
        }
        if (has_fused_parent(node)) {
            if (isSutableChildForFusingSimple(node)) {
                // This feature is disabled to emulate FusingSimple->FusingActivationAndSum->FusingSimple
                // todo: clean all the commented code after benchmark and analysis
//                 // If the node is already marked, it was processed as a child activation in FusingSumActivation
//                if (!fused_tag_is_set(node))
                rt["MayBeFusedInPlugin"] = std::make_shared<VariantWrapper<int64_t>>(VariantWrapper<int64_t>(1));
            } else if (isSutableChildForFusingSumActivation(node)) {
                rt["MayBeFusedInPlugin"] = std::make_shared<VariantWrapper<int64_t>>(VariantWrapper<int64_t>(1));
                // node has only 1 output, because it is Add
                auto child_inputs = node->get_output_target_inputs(0);
                if (child_inputs.size() == 1) {
                    auto child_node = node->get_users()[0];
                    auto &child_rt = child_node->get_rt_info();
                    if (SupportsFusingWithConvolution_SumActivation(child_node)) {
                        child_rt["MayBeFusedInPlugin"] = std::make_shared<VariantWrapper<int64_t>>(
                                VariantWrapper<int64_t>(1));
                    }
//                    else {
//                         // Tag with 0, so this node would not be processed by FusingSimple
//                        child_rt["MayBeFusedInPlugin"] = std::make_shared<VariantWrapper<int64_t>>(VariantWrapper<int64_t>(0));
//                    }
                }
            }
        }
    }
    return true;
}

