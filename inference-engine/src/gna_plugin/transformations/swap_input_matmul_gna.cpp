// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include <memory>
#include <vector>

#include <ngraph/pass/manager.hpp>
#include <ngraph/rt_info.hpp>
#include <numeric>
#include <transformations/swap_input_matmul_gna.hpp>

#include "gna_plugin_log.hpp"

namespace GNAPluginNS {

NGRAPH_RTTI_DEFINITION(SwapInputMatMulFirstInputConstant, "SwapInputMatMulFirstInputConstant", 0);
NGRAPH_RTTI_DEFINITION(SwapInputMatMulSecondInputConstant, "SwapInputMatMulSecondInputConstant", 0);
NGRAPH_RTTI_DEFINITION(SwapInputMatMulWithBias, "SwapInputMatMulWithBias", 0);
NGRAPH_RTTI_DEFINITION(SwapInputMatMulWithFq, "SwapInputMatMulWithFq", 0);

namespace SwapInputMatMul {

void Helper::SwapAndTransposeInputs(
    std::shared_ptr<ngraph::opset8::MatMul> matmul_node,
    std::shared_ptr<ngraph::Node> add,
    std::shared_ptr<ngraph::Node> bias,
    std::shared_ptr<ngraph::Node> fq,
    const std::string& last_layer_name) {
    auto create_transpose =
        [](ngraph::Output<ngraph::Node> node, const std::string& transpose_name) -> std::shared_ptr<ngraph::Node> {
        ngraph::Shape output_shape = node.get_node_shared_ptr()->get_shape();

        std::vector<size_t> transpose_order(output_shape.size());
        std::iota(transpose_order.begin(), transpose_order.end(), 0);
        std::swap(*(transpose_order.end() - 1), *(transpose_order.end() - 2));

        auto transpose = std::make_shared<ngraph::opset8::Transpose>(
                node, ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape {transpose_order.size()}, transpose_order));
        transpose->set_friendly_name(transpose_name);
        return transpose;
    };

    ngraph::NodeVector new_ops;

    gnalog() << "Swap and transpose inputs for " << matmul_node->get_friendly_name() << "\n";
    std::shared_ptr<ngraph::Node> new_matmul = std::make_shared<ngraph::opset8::MatMul>(
        matmul_node->input(1).get_source_output(), matmul_node->input(0).get_source_output(),
        !matmul_node->get_transpose_b(), !matmul_node->get_transpose_a());
    new_matmul->set_friendly_name(matmul_node->get_friendly_name() + "/swap_inputs");
    new_ops.push_back(new_matmul);

    std::shared_ptr<ngraph::Node> old_root_node = matmul_node;
    if (bias != nullptr) {
         // output of MatMul will be transposed comparing with original one, so the bias should be transposed too
         if (bias->get_output_shape(0).size() > 1) {
             bias = create_transpose(bias, bias->get_friendly_name() + "/transpose");
             new_ops.push_back(bias);
         }

         new_matmul = std::make_shared<ngraph::opset8::Add>(new_matmul, bias);
         old_root_node = add;
         new_ops.push_back(new_matmul);
    }

    if (fq != nullptr) {
        new_matmul = fq->clone_with_new_inputs({new_matmul, fq->input_value(1), fq->input_value(2),
            fq->input_value(3), fq->input_value(4)});
        old_root_node = fq;
        new_ops.push_back(new_matmul);
    }

    auto output = create_transpose(new_matmul, last_layer_name);
    new_ops.push_back(output);

    ngraph::copy_runtime_info(matmul_node, new_ops);
    ngraph::replace_node(old_root_node, output);
}

std::shared_ptr<ngraph::Node> Helper::CreateMatmul(
    bool is_first_constant,
    ngraph::pattern::op::ValuePredicate const_predicate,
    ngraph::pattern::op::ValuePredicate matmul_predicate) {
    auto constant = ngraph::pattern::wrap_type<ngraph::opset8::Constant>({}, const_predicate);
    auto fake_quantize = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({constant,
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>()});
    auto matmul_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{constant, fake_quantize});
    if (is_first_constant) {
        return ngraph::pattern::wrap_type<ngraph::opset8::MatMul>(
            {matmul_input, ngraph::pattern::any_input()}, matmul_predicate);
    }
    return ngraph::pattern::wrap_type<ngraph::opset8::MatMul>(
        {ngraph::pattern::any_input(), matmul_input}, matmul_predicate);
}

} // namespace SwapInputMatMul

SwapInputMatMulFirstInputConstant::SwapInputMatMulFirstInputConstant() {
    MATCHER_SCOPE(SwapInputMatMulFirstInputConstant);
    std::shared_ptr<ngraph::pattern::Matcher> matcher;
    ngraph::graph_rewrite_callback callback;
    IE_ASSERT(SwapInputMatMul::Helper::CreateMatcher<SwapInputMatMulFirstInputConstant>(matcher, callback));
    this->register_matcher(matcher, callback);
}

SwapInputMatMulSecondInputConstant::SwapInputMatMulSecondInputConstant() {
    MATCHER_SCOPE(SwapInputMatMulSecondInputConstant);
    std::shared_ptr<ngraph::pattern::Matcher> matcher;
    ngraph::graph_rewrite_callback callback;
    IE_ASSERT(SwapInputMatMul::Helper::CreateMatcher<SwapInputMatMulSecondInputConstant>(matcher, callback));
    this->register_matcher(matcher, callback);
}

SwapInputMatMulWithBias::SwapInputMatMulWithBias() {
    MATCHER_SCOPE(SwapInputMatMulWithBias);
    std::shared_ptr<ngraph::pattern::Matcher> matcher;
    ngraph::graph_rewrite_callback callback;
    IE_ASSERT(SwapInputMatMul::Helper::CreateMatcher<SwapInputMatMulWithBias>(matcher, callback));
    this->register_matcher(matcher, callback);
}

SwapInputMatMulWithFq::SwapInputMatMulWithFq() {
    MATCHER_SCOPE(SwapInputMatMulWithFq);
    std::shared_ptr<ngraph::pattern::Matcher> matcher;
    ngraph::graph_rewrite_callback callback;
    IE_ASSERT(SwapInputMatMul::Helper::CreateMatcher<SwapInputMatMulWithFq>(matcher, callback));
    this->register_matcher(matcher, callback);
}
} // namespace GNAPluginNS
