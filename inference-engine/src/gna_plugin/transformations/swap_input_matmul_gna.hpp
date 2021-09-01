// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ie/ie_common.h>

namespace GNAPluginNS {

class SwapInputMatMulFirstInputConstant;
class SwapInputMatMulSecondInputConstant;
class SwapInputMatMulWithBias;
class SwapInputMatMulWithFq;

namespace SwapInputMatMul {
struct Helper {
    static void SwapAndTransposeInputs(
        std::shared_ptr<ngraph::opset8::MatMul> matmul_node,
        std::shared_ptr<ngraph::Node> add,
        std::shared_ptr<ngraph::Node> bias,
        std::shared_ptr<ngraph::Node> fq,
        const std::string& last_layer_name);
    static std::shared_ptr<ngraph::Node> CreateMatmul(
        bool is_first_constant,
        ngraph::pattern::op::ValuePredicate const_predicate,
        ngraph::pattern::op::ValuePredicate matmul_predicate = ngraph::pattern::has_static_shape());

    template<typename T>
    static bool CreateMatcher(std::shared_ptr<ngraph::pattern::Matcher>& matcher, ngraph::graph_rewrite_callback& callback) {
        auto matmul1 = CreateMatmul(true, [](const ngraph::Output<ngraph::Node>& node) {
            auto shape = node.get_node_shared_ptr()->get_output_shape(0);
            return shape.size() == 2 && shape[0] > 8 && shape[0] % 8 == 0 && shape[1] % 8 == 0; });
        if (std::is_same<T, SwapInputMatMulFirstInputConstant>::value) {
            callback = [=](ngraph::pattern::Matcher& m) {
                const auto& pattern_map = m.get_pattern_value_map();
                auto matmul_node = std::dynamic_pointer_cast<ngraph::opset8::MatMul>(pattern_map.at(matmul1).get_node_shared_ptr());
                IE_ASSERT(matmul_node != nullptr);
                Helper::SwapAndTransposeInputs(matmul_node, nullptr, nullptr, nullptr, "");
                return true;
            };
            matcher = std::make_shared<ngraph::pattern::Matcher>(matmul1, T::Name());
            return true;
        }

        auto matmul2 = CreateMatmul(
            false,
            [](const ngraph::Output<ngraph::Node>& node) { return true; },
            [](const ngraph::Output<ngraph::Node>& node) {
                auto input_shape = node.get_node_shared_ptr()->get_input_shape(0);
                input_shape.erase(std::remove(input_shape.begin(), input_shape.end(), 1), input_shape.end());
                auto constant_shape = node.get_node_shared_ptr()->get_input_shape(1);
                return node.get_partial_shape().is_static() &&
                    constant_shape.size() == 2 &&
                    (constant_shape[0] <= 8 || constant_shape[1] <= 8) &&
                    input_shape.size() == 2 &&
                    input_shape[0] > 8; });
        if (std::is_same<T, SwapInputMatMulSecondInputConstant>::value) {
            callback = [=](ngraph::pattern::Matcher& m) {
                const auto& pattern_map = m.get_pattern_value_map();
                auto matmul_node = std::dynamic_pointer_cast<ngraph::opset8::MatMul>(pattern_map.at(matmul2).get_node_shared_ptr());
                IE_ASSERT(matmul_node != nullptr);
                Helper::SwapAndTransposeInputs(matmul_node, nullptr, nullptr, nullptr, "");
                return true;
            };
            matcher = std::make_shared<ngraph::pattern::Matcher>(matmul2, T::Name());
            return true;
        }

        auto bias = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
        auto add_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{matmul1, matmul2});
        auto add = ngraph::pattern::wrap_type<ngraph::opset8::Add>({add_input, bias});
        if (std::is_same<T, SwapInputMatMulWithBias>::value) {
            callback = [=](ngraph::pattern::Matcher& m) {
                const auto& pattern_map = m.get_pattern_value_map();
                auto iter = pattern_map.find(matmul1);
                if (iter == pattern_map.end()) {
                    iter = pattern_map.find(matmul2);
                    if (iter == pattern_map.end()) {
                        return false;
                    }
                }

                auto matmul_node = std::dynamic_pointer_cast<ngraph::opset8::MatMul>(iter->second.get_node_shared_ptr());
                IE_ASSERT(matmul_node != nullptr);
                Helper::SwapAndTransposeInputs(
                    matmul_node,
                    pattern_map.at(add).get_node_shared_ptr(),
                    pattern_map.at(bias).get_node_shared_ptr(),
                    nullptr,
                    pattern_map.at(add).get_node_shared_ptr()->get_friendly_name());
                return true;
            };
            matcher = std::make_shared<ngraph::pattern::Matcher>(add, T::Name());
            return true;
        }

        auto fq_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{add, add_input});
        auto fq = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({fq_input,
            ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
            ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
            ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
            ngraph::pattern::wrap_type<ngraph::opset8::Constant>()});
        if (std::is_same<T, SwapInputMatMulWithFq>::value) {
            callback = [=](ngraph::pattern::Matcher& m) {
                const auto& pattern_map = m.get_pattern_value_map();
                auto iter = pattern_map.find(matmul1);
                if (iter == pattern_map.end()) {
                    iter = pattern_map.find(matmul2);
                    if (iter == pattern_map.end()) {
                        return false;
                    }
                }

                auto iter_add = pattern_map.find(add);
                auto iter_bias = pattern_map.find(bias);
                auto matmul_node = std::dynamic_pointer_cast<ngraph::opset8::MatMul>(iter->second.get_node_shared_ptr());
                IE_ASSERT(matmul_node != nullptr);
                Helper::SwapAndTransposeInputs(
                    matmul_node,
                    iter_add != pattern_map.end() ? iter_add->second.get_node_shared_ptr() : nullptr,
                    iter_bias != pattern_map.end() ? iter_bias->second.get_node_shared_ptr() : nullptr,
                    pattern_map.at(fq).get_node_shared_ptr(),
                    pattern_map.at(fq).get_node_shared_ptr()->get_friendly_name());
                return true;
            };
            matcher = std::make_shared<ngraph::pattern::Matcher>(fq, T::Name());
            return true;
        }

        return false;
    }
}; // struct Helper

} // namespace SwapInputMatMul

// @brief Swaps and transposes inputs of MatMul if its first input is const and its batch size isn't supported by GNA
class SwapInputMatMulFirstInputConstant: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SwapInputMatMulFirstInputConstant();
    static std::string Name() { return "SwapInputMatMulFirstInputConstant"; }
};

// @brief Swaps and transposes inputs of MatMul if
// 1. its first input non-const and its batch size isn't supported by GNA
// 2. its second input is const and its st input batch size less than or equal to 8
class SwapInputMatMulSecondInputConstant: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SwapInputMatMulSecondInputConstant();
    static std::string Name() { return "SwapInputMatMulSecondInputConstant"; }
};

class SwapInputMatMulWithBias: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SwapInputMatMulWithBias();
    static std::string Name() { return "SwapInputMatMulWithBias"; }
};

class SwapInputMatMulWithFq: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SwapInputMatMulWithFq();
    static std::string Name() { return "SwapInputMatMulWithFq"; }
};
} // namespace GNAPluginNS
