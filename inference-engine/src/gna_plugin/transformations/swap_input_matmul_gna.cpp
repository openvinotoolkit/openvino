// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include <ngraph/pass/manager.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <numeric>
#include <transformations/swap_input_matmul_gna.hpp>

#include "gna_plugin_log.hpp"

using namespace GNAPluginNS;

NGRAPH_RTTI_DEFINITION(SwapInputMatMul, "SwapInputMatMul", 0);
NGRAPH_RTTI_DEFINITION(SwapInputMatMulWithBias, "SwapInputMatMulWithBias", 0);
NGRAPH_RTTI_DEFINITION(SwapInputMatMulWithFq, "SwapInputMatMulWithFq", 0);

static void SwapAndTransposeInputs(std::shared_ptr<ngraph::opset7::MatMul> matmul_node,
                                   std::shared_ptr<ngraph::Node> add,
                                   std::shared_ptr<ngraph::Node> bias,
                                   std::shared_ptr<ngraph::Node> fq) {
    auto create_transpose =
        [](ngraph::Output<ngraph::Node> node, const std::string& transpose_name) -> std::shared_ptr<ngraph::Node> {
        ngraph::Shape output_shape = node.get_node_shared_ptr()->get_shape();

        std::vector<size_t> transpose_order(output_shape.size());
        std::iota(transpose_order.begin(), transpose_order.end(), 0);
        std::swap(*(transpose_order.end() - 1), *(transpose_order.end() - 2));

        auto transpose = std::make_shared<ngraph::opset7::Transpose>(
                node, ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape {transpose_order.size()}, transpose_order));
        transpose->set_friendly_name(transpose_name);
        return transpose;
    };

    ngraph::NodeVector new_ops;

    gnalog() << "Swap and transpose inputs for " << matmul_node->get_friendly_name() << "\n";
    std::shared_ptr<ngraph::Node> new_matmul = std::make_shared<ngraph::opset7::MatMul>(
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

         new_matmul = std::make_shared<ngraph::opset7::Add>(new_matmul, bias);
         old_root_node = add;
         new_ops.push_back(new_matmul);
    }

    if (fq != nullptr) {
        new_matmul = fq->clone_with_new_inputs({new_matmul, fq->input_value(1), fq->input_value(2),
            fq->input_value(3), fq->input_value(4)});
        old_root_node = fq;
        new_ops.push_back(new_matmul);
    }

    auto output = create_transpose(new_matmul,  matmul_node->get_friendly_name());
    new_ops.push_back(output);

    ngraph::copy_runtime_info(matmul_node, new_ops);
    ngraph::replace_node(old_root_node, output);
}

SwapInputMatMul::SwapInputMatMul() {
    auto constant = ngraph::pattern::wrap_type<ngraph::opset7::Constant>({}, [](const ngraph::Output<ngraph::Node>& node) {
        auto shape = node.get_node_shared_ptr()->get_output_shape(0);
        if (shape.size() != 2 || shape[0] < 8 || ((shape[0] % 8 != 0 || shape[1] % 8 != 0))) {
            return false;
        }
        return true;
    });
    auto fake_quantize = ngraph::pattern::wrap_type<ngraph::opset7::FakeQuantize>({constant,
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>()});
    auto matmul_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{constant, fake_quantize});
    auto matmul = ngraph::pattern::wrap_type<ngraph::opset7::MatMul>({matmul_input, ngraph::pattern::any_input()},
                                                                      ngraph::pattern::has_static_shape());
    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto matmul_node = std::dynamic_pointer_cast<ngraph::opset7::MatMul>(pattern_map.at(matmul).get_node_shared_ptr());
        IE_ASSERT(matmul_node != nullptr);
        SwapAndTransposeInputs(matmul_node, nullptr, nullptr, nullptr);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matmul, "SwapInputMatMul");
    this->register_matcher(m, callback);
}

SwapInputMatMulWithBias::SwapInputMatMulWithBias() {
    auto constant = ngraph::pattern::wrap_type<ngraph::opset7::Constant>({}, [](const ngraph::Output<ngraph::Node>& node) {
        auto shape = node.get_node_shared_ptr()->get_output_shape(0);
        if (shape.size() != 2 || shape[0] < 8 || ((shape[0] % 8 != 0 || shape[1] % 8 != 0))) {
            return false;
        }
        return true;
    });
    auto fake_quantize = ngraph::pattern::wrap_type<ngraph::opset7::FakeQuantize>({constant,
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>()});
    auto matmul_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{constant, fake_quantize});
    auto matmul = ngraph::pattern::wrap_type<ngraph::opset7::MatMul>({matmul_input, ngraph::pattern::any_input()},
                                                                      ngraph::pattern::has_static_shape());
    auto bias = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto add = ngraph::pattern::wrap_type<ngraph::opset7::Add>({matmul, bias});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto matmul_node = std::dynamic_pointer_cast<ngraph::opset7::MatMul>(pattern_map.at(matmul).get_node_shared_ptr());
        IE_ASSERT(matmul_node != nullptr);
        SwapAndTransposeInputs(matmul_node, pattern_map.at(add).get_node_shared_ptr(),
            pattern_map.at(bias).get_node_shared_ptr(), nullptr);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(add, "SwapInputMatMulWithBias");
    this->register_matcher(m, callback);
}

SwapInputMatMulWithFq::SwapInputMatMulWithFq() {
    auto constant = ngraph::pattern::wrap_type<ngraph::opset7::Constant>({}, [](const ngraph::Output<ngraph::Node>& node) {
        auto shape = node.get_node_shared_ptr()->get_output_shape(0);
        if (shape.size() != 2 || shape[0] < 8 || ((shape[0] % 8 != 0 || shape[1] % 8 != 0))) {
            return false;
        }
        return true;
    });
    auto fake_quantize = ngraph::pattern::wrap_type<ngraph::opset7::FakeQuantize>({constant,
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>()});
    auto matmul_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{constant, fake_quantize});
    auto matmul = ngraph::pattern::wrap_type<ngraph::opset7::MatMul>({matmul_input, ngraph::pattern::any_input()},
                                                                      ngraph::pattern::has_static_shape());
    auto bias = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto add = ngraph::pattern::wrap_type<ngraph::opset7::Add>({matmul, bias});
    auto matmul_out = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{add, matmul});
    auto out_fq = ngraph::pattern::wrap_type<ngraph::opset7::FakeQuantize>({matmul_out,
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>()});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto matmul_node = std::dynamic_pointer_cast<ngraph::opset7::MatMul>(pattern_map.at(matmul).get_node_shared_ptr());
        IE_ASSERT(matmul_node != nullptr);
        auto add_it = pattern_map.find(add);
        auto add_node = (add_it == std::end(pattern_map) ? nullptr : add_it->second.get_node_shared_ptr());
        auto bias_it = pattern_map.find(bias);
        auto bias_node = (bias_it == std::end(pattern_map) ? nullptr : bias_it->second.get_node_shared_ptr());
        SwapAndTransposeInputs(matmul_node, add_node, bias_node, pattern_map.at(out_fq).get_node_shared_ptr());
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(out_fq, "SwapInputMatMulWithFq");
    this->register_matcher(m, callback);
}