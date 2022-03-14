// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/swap_input_matmul_gna.hpp>
#include <openvino/cc/ngraph/itt.hpp>

#include <memory>
#include <vector>

#include <ngraph/pass/manager.hpp>
#include <ngraph/rt_info.hpp>
#include <numeric>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ie/ie_common.h>

#include "gna_plugin_log.hpp"

namespace GNAPluginNS {

static void SwapAndTransposeInputs(
    std::shared_ptr<ngraph::opset8::MatMul> matmul_node,
    const std::string& last_layer_name,
    std::shared_ptr<ngraph::Node> add = nullptr,
    std::shared_ptr<ngraph::Node> bias = nullptr,
    std::shared_ptr<ngraph::Node> fq = nullptr,
    std::shared_ptr<ngraph::Node> act = nullptr,
    std::shared_ptr<ngraph::Node> transpose = nullptr) {
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

    auto transpose_matmul_input = [matmul_node, &new_ops, create_transpose](size_t ix) {
        std::shared_ptr<ngraph::Node> matmul_input = matmul_node->input_value(ix).get_node_shared_ptr();
        auto input_transpose = std::dynamic_pointer_cast<ngraph::opset8::Transpose>(matmul_input);
        if (input_transpose != nullptr) {
            matmul_input = matmul_input->input_value(0).get_node_shared_ptr();
            ngraph::replace_output_update_name(input_transpose->output(0), input_transpose->input_value(0));
        } else {
            matmul_input = create_transpose(matmul_node->input_value(ix), matmul_node->get_friendly_name() + "/input_transpose");
            new_ops.push_back(matmul_input);
        }
        return matmul_input;
    };

    gnalog() << "Swap and transpose inputs for " << matmul_node->get_friendly_name() << "\n";

    bool first_input_const = false;
    bool second_input_const = false;
    auto first_input = matmul_node->input_value(0).get_node_shared_ptr();
    auto second_input = matmul_node->input_value(1).get_node_shared_ptr();
    if (std::dynamic_pointer_cast<ngraph::opset8::FakeQuantize>(first_input)) {
        first_input = first_input->input_value(0).get_node_shared_ptr();
    }
    if (std::dynamic_pointer_cast<ngraph::opset8::FakeQuantize>(second_input)) {
        second_input = second_input->input_value(1).get_node_shared_ptr();
    }
    if (std::dynamic_pointer_cast<ngraph::opset8::Constant>(first_input)) {
        first_input_const = true;
    }
    if (std::dynamic_pointer_cast<ngraph::opset8::Constant>(second_input)) {
        second_input_const = true;
    }

    auto input1 = (!first_input_const && second_input_const) ? matmul_node->input_value(1) : transpose_matmul_input(1);
    auto input2 = first_input_const ? matmul_node->input_value(0) : transpose_matmul_input(0);
    bool transpose_1 = (!first_input_const && second_input_const) ? !matmul_node->get_transpose_b() : matmul_node->get_transpose_b();
    bool transpose_2 = first_input_const ? !matmul_node->get_transpose_a() : matmul_node->get_transpose_a();
    std::shared_ptr<ngraph::Node> new_node = std::make_shared<ngraph::opset8::MatMul>(input1, input2, transpose_1, transpose_2);
    new_node->set_friendly_name(matmul_node->get_friendly_name() + "/swap_inputs");
    new_ops.push_back(new_node);

    std::shared_ptr<ngraph::Node> old_root_node = matmul_node;
    if (bias != nullptr) {
        // output of MatMul will be transposed comparing with original one, so the bias should be transposed too
        if (bias->get_output_shape(0).size() > 1) {
            bias = create_transpose(bias, bias->get_friendly_name() + "/transpose");
            new_ops.push_back(bias);

            auto transpose_shape = bias->get_output_shape(0);
            auto matmul_shape = matmul_node->get_output_shape(0);
            if (transpose_shape.size() > matmul_shape.size()) {
                std::vector<size_t> reshape_shape(matmul_shape.size(), 1);
                std::copy_if(transpose_shape.begin(), transpose_shape.end(), reshape_shape.begin(), [](size_t e) { return e > 1; });
                bias = std::make_shared<ngraph::opset8::Reshape>(bias,
                    std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::i64,
                        ngraph::Shape{reshape_shape.size()}, reshape_shape), false);
                bias->set_friendly_name(add->get_friendly_name() + "/reshape");
                ngraph::copy_runtime_info(add, bias);
                new_ops.push_back(bias);
            }
        }

        new_node = std::make_shared<ngraph::opset8::Add>(new_node, bias);
        old_root_node = add;
        new_ops.push_back(new_node);
    }

    if (fq != nullptr) {
        new_node = fq->clone_with_new_inputs({new_node, fq->input_value(1), fq->input_value(2),
            fq->input_value(3), fq->input_value(4)});
        old_root_node = fq;
        new_ops.push_back(new_node);
    }

    if (act != nullptr) {
        new_node = act->clone_with_new_inputs({new_node});
        old_root_node = act;
        new_ops.push_back(new_node);
    }

    if (transpose == nullptr) {
        new_node = create_transpose(new_node, last_layer_name);
        new_ops.push_back(new_node);
    } else {
        ngraph::replace_output_update_name(transpose->output(0), transpose->input_value(0));
        new_node->set_friendly_name(last_layer_name);
    }

    ngraph::copy_runtime_info(matmul_node, new_ops);
    ngraph::replace_node(old_root_node, new_node);
}

static std::shared_ptr<ngraph::Node> CreateMatmul(
    bool is_first_constant,
    ngraph::pattern::op::ValuePredicate const_predicate,
    ngraph::pattern::op::ValuePredicate matmul_predicate = ngraph::pattern::has_static_shape()) {
    auto constant = ngraph::pattern::wrap_type<ngraph::opset8::Constant>({}, const_predicate);
    auto fake_quantize = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({constant,
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>()});
    auto matmul_const_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{constant, fake_quantize});
    auto transpose = ngraph::pattern::wrap_type<ngraph::opset8::Transpose>();
    auto matmul_non_const_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{transpose,
        ngraph::pattern::any_input()});
    if (is_first_constant) {
        return ngraph::pattern::wrap_type<ngraph::opset8::MatMul>(
            {matmul_const_input, matmul_non_const_input}, matmul_predicate);
    }
    return ngraph::pattern::wrap_type<ngraph::opset8::MatMul>(
        {ngraph::pattern::any_input(), matmul_non_const_input}, matmul_predicate);
}

static std::shared_ptr<ngraph::Node> CreateMatmuls(
    std::shared_ptr<ngraph::Node>& matmul1,
    std::shared_ptr<ngraph::Node>& matmul2) {
    matmul1 = CreateMatmul(
        true,
        [](const ngraph::Output<ngraph::Node>& node) { return true; },
        [](const ngraph::Output<ngraph::Node>& node) {
            auto matmul_node = std::dynamic_pointer_cast<ngraph::opset8::MatMul>(node.get_node_shared_ptr());
            IE_ASSERT(matmul_node != nullptr);
            auto input_shape = matmul_node->get_input_shape(0);
            return input_shape.size() == 2 &&
                (!matmul_node->get_transpose_a() && input_shape[0] > 8 ||
                matmul_node->get_transpose_a() && input_shape[1] > 8); });
    matmul2 = CreateMatmul(
        false,
        [](const ngraph::Output<ngraph::Node>& node) { return true; },
        [](const ngraph::Output<ngraph::Node>& node) {
            auto matmul_node = std::dynamic_pointer_cast<ngraph::opset8::MatMul>(node.get_node_shared_ptr());
            IE_ASSERT(matmul_node != nullptr);
            auto first_input_shape = matmul_node->get_input_shape(0);
            first_input_shape.erase(std::remove(first_input_shape.begin(), first_input_shape.end(), 1), first_input_shape.end());
            auto second_input_shape = matmul_node->get_input_shape(1);
            return node.get_partial_shape().is_static() &&
                second_input_shape.size() == 2 &&
                (!matmul_node->get_transpose_b() && second_input_shape[1] <= 8 ||
                 matmul_node->get_transpose_b() && second_input_shape[0] <= 8) &&
                first_input_shape.size() == 2 &&
                first_input_shape[0] > 8; });
    return std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{matmul1, matmul2});
}

SwapInputMatMul::SwapInputMatMul() {
    MATCHER_SCOPE(SwapInputMatMul);
    std::shared_ptr<ngraph::Node> matmul1;
    std::shared_ptr<ngraph::Node> matmul2;
    auto matmul = CreateMatmuls(matmul1, matmul2);
    auto callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto iter = pattern_map.find(matmul1);
        if (iter == pattern_map.end() && (iter = pattern_map.find(matmul2)) == pattern_map.end()) {
            return false;
        }

        auto matmul_node = std::dynamic_pointer_cast<ngraph::opset8::MatMul>(iter->second.get_node_shared_ptr());
        IE_ASSERT(matmul_node != nullptr);
        SwapAndTransposeInputs(matmul_node, matmul_node->get_friendly_name());
        return true;
    };

    auto matcher = std::make_shared<ngraph::pattern::Matcher>(matmul, "SwapInputMatMul");
    this->register_matcher(matcher, callback);
}

SwapInputMatMulWithBias::SwapInputMatMulWithBias() {
    MATCHER_SCOPE(SwapInputMatMulWithBias);
    std::shared_ptr<ngraph::Node> matmul1;
    std::shared_ptr<ngraph::Node> matmul2;
    auto matmul = CreateMatmuls(matmul1, matmul2);
    auto bias = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto add = ngraph::pattern::wrap_type<ngraph::opset8::Add>({matmul, bias});
    auto callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto iter = pattern_map.find(matmul1);
        if (iter == pattern_map.end() && (iter = pattern_map.find(matmul2)) == pattern_map.end()) {
            return false;
        }

        auto matmul_node = std::dynamic_pointer_cast<ngraph::opset8::MatMul>(iter->second.get_node_shared_ptr());
        IE_ASSERT(matmul_node != nullptr);
        SwapAndTransposeInputs(
            matmul_node,
            pattern_map.at(add).get_node_shared_ptr()->get_friendly_name(),
            pattern_map.at(add).get_node_shared_ptr(),
            pattern_map.at(bias).get_node_shared_ptr());
        return true;
    };

    auto matcher = std::make_shared<ngraph::pattern::Matcher>(add, "SwapInputMatMulWithBias");
    this->register_matcher(matcher, callback);
}

SwapInputMatMulWithFq::SwapInputMatMulWithFq() {
    MATCHER_SCOPE(SwapInputMatMulWithFq);
    std::shared_ptr<ngraph::Node> matmul1;
    std::shared_ptr<ngraph::Node> matmul2;
    auto matmul = CreateMatmuls(matmul1, matmul2);
    auto bias = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto add = ngraph::pattern::wrap_type<ngraph::opset8::Add>({matmul, bias});
    auto fq_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{add, matmul});
    auto fq = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({fq_input,
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>()});
    auto callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto iter = pattern_map.find(matmul1);
        if (iter == pattern_map.end() && (iter = pattern_map.find(matmul2)) == pattern_map.end()) {
            return false;
        }

        auto iter_add = pattern_map.find(add);
        auto iter_bias = pattern_map.find(bias);
        auto matmul_node = std::dynamic_pointer_cast<ngraph::opset8::MatMul>(iter->second.get_node_shared_ptr());
        IE_ASSERT(matmul_node != nullptr);
        SwapAndTransposeInputs(
            matmul_node,
            pattern_map.at(fq).get_node_shared_ptr()->get_friendly_name(),
            iter_add != pattern_map.end() ? iter_add->second.get_node_shared_ptr() : nullptr,
            iter_bias != pattern_map.end() ? iter_bias->second.get_node_shared_ptr() : nullptr,
            pattern_map.at(fq).get_node_shared_ptr());
        return true;
    };

    auto matcher = std::make_shared<ngraph::pattern::Matcher>(fq, "SwapInputMatMulWithFq");
    this->register_matcher(matcher, callback);
}

SwapInputMatMulWithAct::SwapInputMatMulWithAct() {
    MATCHER_SCOPE(SwapInputMatMulWithAct);
    std::shared_ptr<ngraph::Node> matmul1;
    std::shared_ptr<ngraph::Node> matmul2;
    auto matmul = CreateMatmuls(matmul1, matmul2);
    auto bias = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto add = ngraph::pattern::wrap_type<ngraph::opset8::Add>({matmul, bias});
    auto fq_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{add, matmul});
    auto fq = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({fq_input,
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>()});
    auto act_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{fq_input, fq});
    auto act = ngraph::pattern::wrap_type<ngraph::opset8::Relu, ngraph::opset8::Sigmoid,
        ngraph::opset8::Tanh, ngraph::opset8::Abs, ngraph::opset8::Log, ngraph::opset8::Exp,
        ngraph::opset8::Sign, ngraph::opset8::Clamp>({act_input});
    auto callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto iter = pattern_map.find(matmul1);
        if (iter == pattern_map.end() && (iter = pattern_map.find(matmul2)) == pattern_map.end()) {
            return false;
        }

        auto iter_add = pattern_map.find(add);
        auto iter_bias = pattern_map.find(bias);
        auto iter_fq = pattern_map.find(fq);
        auto matmul_node = std::dynamic_pointer_cast<ngraph::opset8::MatMul>(iter->second.get_node_shared_ptr());
        IE_ASSERT(matmul_node != nullptr);
        SwapAndTransposeInputs(
            matmul_node,
            pattern_map.at(act).get_node_shared_ptr()->get_friendly_name(),
            iter_add != pattern_map.end() ? iter_add->second.get_node_shared_ptr() : nullptr,
            iter_bias != pattern_map.end() ? iter_bias->second.get_node_shared_ptr() : nullptr,
            iter_fq != pattern_map.end() ? iter_fq->second.get_node_shared_ptr() : nullptr,
            pattern_map.at(act).get_node_shared_ptr());
        return true;
    };

    auto matcher = std::make_shared<ngraph::pattern::Matcher>(act, "SwapInputMatMulWithAct");
    this->register_matcher(matcher, callback);
}

SwapInputMatMulWithTrailingTranspose::SwapInputMatMulWithTrailingTranspose() {
    MATCHER_SCOPE(SwapInputMatMulWithTrailingTranspose);
    std::shared_ptr<ngraph::Node> matmul1;
    std::shared_ptr<ngraph::Node> matmul2;
    auto matmul = CreateMatmuls(matmul1, matmul2);
    auto bias = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto add = ngraph::pattern::wrap_type<ngraph::opset8::Add>({matmul, bias});
    auto fq_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{add, matmul});
    auto fq = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({fq_input,
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>()});
    auto act_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{fq_input, fq});
    auto act = ngraph::pattern::wrap_type<ngraph::opset8::Relu, ngraph::opset8::Sigmoid,
        ngraph::opset8::Tanh, ngraph::opset8::Abs, ngraph::opset8::Log, ngraph::opset8::Exp,
        ngraph::opset8::Sign, ngraph::opset8::Clamp>({act_input});
    auto transpose_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{act_input, act});
    auto transpose = ngraph::pattern::wrap_type<ngraph::opset8::Transpose>({transpose_input, ngraph::pattern::any_input()});
    auto callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto iter = pattern_map.find(matmul1);
        if (iter == pattern_map.end() && (iter = pattern_map.find(matmul2)) == pattern_map.end()) {
            return false;
        }

        auto iter_add = pattern_map.find(add);
        auto iter_bias = pattern_map.find(bias);
        auto iter_fq = pattern_map.find(fq);
        auto iter_act = pattern_map.find(act);
        auto matmul_node = std::dynamic_pointer_cast<ngraph::opset8::MatMul>(iter->second.get_node_shared_ptr());
        IE_ASSERT(matmul_node != nullptr);
        SwapAndTransposeInputs(
            matmul_node,
            pattern_map.at(transpose).get_node_shared_ptr()->get_friendly_name(),
            iter_add != pattern_map.end() ? iter_add->second.get_node_shared_ptr() : nullptr,
            iter_bias != pattern_map.end() ? iter_bias->second.get_node_shared_ptr() : nullptr,
            iter_fq != pattern_map.end() ? iter_fq->second.get_node_shared_ptr() : nullptr,
            iter_act != pattern_map.end() ? iter_act->second.get_node_shared_ptr() : nullptr,
            pattern_map.at(transpose).get_node_shared_ptr());
        return true;
    };

    auto matcher = std::make_shared<ngraph::pattern::Matcher>(transpose, "SwapInputMatMulWithTrailingTranspose");
    this->register_matcher(matcher, callback);
}
} // namespace GNAPluginNS
