// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/insert_reshape_around_matmul.hpp"
#include <openvino/cc/ngraph/itt.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>
#include <ie/ie_common.h>

#include "gna_plugin_log.hpp"

namespace GNAPluginNS {

static bool InsertReshape(
    ngraph::pattern::Matcher &matcher,
    const std::shared_ptr<ngraph::Node>& input,
    const std::shared_ptr<ngraph::Node>& matmul1,
    const std::shared_ptr<ngraph::Node>& matmul2,
    const std::shared_ptr<ngraph::Node>& add1 = nullptr,
    const std::shared_ptr<ngraph::Node>& add2 = nullptr,
    const std::shared_ptr<ngraph::Node>& fake_quantize = nullptr,
    const std::shared_ptr<ngraph::Node>& transpose = nullptr) {
    const auto& pattern_map = matcher.get_pattern_value_map();
    size_t matmul_input_index = 1;
    auto iter = pattern_map.find(matmul1);
    if (iter == pattern_map.end()) {
        iter = pattern_map.find(matmul2);
        if ((iter = pattern_map.find(matmul2)) == pattern_map.end()) {
            return false;
        }

        matmul_input_index = 0;
    }

    std::shared_ptr<ngraph::Node> matmul_node = iter->second.get_node_shared_ptr();
    if ((iter = pattern_map.find(input)) == std::end(pattern_map)) {
        return false;
    }

    auto first_node = iter->second.get_node_shared_ptr();
    size_t add_input_index = 0;
    iter = pattern_map.find(add1);
    std::shared_ptr<ngraph::Node> add_node = nullptr;
    if (iter != pattern_map.end()) {
        add_node = iter->second.get_node_shared_ptr();
        add_input_index = std::dynamic_pointer_cast<ngraph::opset8::MatMul>(add_node->get_input_node_shared_ptr(0)) ? 1 : 0;
    }

    // If there is an Add layer, check if it doesn't require inserting a Reshape
    // to align its dimensions with reshaped MatMul's dimensions
    if (add_node) {
        auto add_input = add_node->get_input_node_shared_ptr(add_input_index);
        if (add_input->get_output_shape(0).size() != 2) {
            auto consumers = add_input->output(0).get_target_inputs();
            std::vector<int> before_shape = {-1, static_cast<int>(add_input->get_output_shape(0).back())};
            auto reshape_add_input = ngraph::op::util::make_try_fold<ngraph::opset8::Reshape>(add_input,
                std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{before_shape.size()}, before_shape), false);
            reshape_add_input->set_friendly_name(reshape_add_input->get_friendly_name() + "/reshape_before_add");
            ngraph::copy_runtime_info(add_node, reshape_add_input);
            for (auto consumer : consumers) {
                consumer.replace_source_output(reshape_add_input);
            }
        }
    }

    std::vector<std::shared_ptr<ngraph::Node>> nodes = { matmul_node };
    for (auto node : {add2, add1, fake_quantize, transpose}) {
        iter = pattern_map.find(node);
        if (iter != pattern_map.end()) {
            nodes.push_back(iter->second.get_node_shared_ptr());
        }
    }

    auto last_node_shape = nodes.back()->get_output_shape(0);
    auto reshape_input_node = std::dynamic_pointer_cast<ngraph::opset8::Reshape>(first_node);
    bool need_reshape_before = !reshape_input_node || reshape_input_node->get_output_shape(0).size() != 2;
    if (need_reshape_before) {
        std::vector<int> before_shape = {-1, static_cast<int>(first_node->get_output_shape(0).back())};
        auto reshape_before_node = std::make_shared<ngraph::opset8::Reshape>(first_node,
            std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{before_shape.size()}, before_shape), false);
        reshape_before_node->set_friendly_name(matmul_node->get_friendly_name() + "/reshape_before_matmul");
        ngraph::copy_runtime_info(first_node, reshape_before_node);
        matmul_node->input(matmul_input_index).replace_source_output(reshape_before_node->output(0));
        if (auto transpose_node = std::dynamic_pointer_cast<ngraph::opset8::Transpose>(nodes.back())) {
            nodes.pop_back();
            std::reverse(nodes.begin(), nodes.end());
            while (!nodes.empty()) {
                auto node_copy = nodes.back()->clone_with_new_inputs(nodes.back()->input_values());
                ngraph::copy_runtime_info(nodes.back(), node_copy);
                ngraph::replace_node(nodes.back(), node_copy);
                nodes.pop_back();
            }

            auto transpose_input_shape = transpose_node->input_values()[0].get_node_shared_ptr()->get_output_shape(0);
            auto transpose_constant_shape = transpose_node->input_values()[1].get_node_shared_ptr()->get_output_shape(0);
            if (std::count_if(transpose_input_shape.begin(), transpose_input_shape.end(), [](size_t n) { return n > 1; }) > 2) {
                THROW_GNA_EXCEPTION << "The number of dimensions that are greater than 1 is greater than 2"
                    << " for Transpose layer (" << transpose_node->get_friendly_name() << ")."
                    << " For this reason, there is no way to determine permutation shape.";
            }
            std::vector<int> permutation_shape = {1, 0};
            auto transpose_node_copy = transpose_node->clone_with_new_inputs(
                {transpose_node->input_values()[0],
                std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::i64,
                    ngraph::Shape{permutation_shape.size()}, permutation_shape)});
            ngraph::copy_runtime_info(transpose_node, transpose_node_copy);
            ngraph::replace_node(transpose_node, transpose_node_copy);
            nodes.push_back(transpose_node_copy);
        }
    }

    auto consumers = nodes.back()->output(0).get_target_inputs();
    bool need_reshape_after = false;
    for (auto consumer : consumers) {
        auto reshape_output_node = dynamic_cast<ngraph::opset8::Reshape*>(consumer.get_node());
        if (!reshape_output_node || reshape_output_node->get_output_shape(0).size() != last_node_shape.size()) {
            need_reshape_after = true;
            break;
        }
    }

    if (need_reshape_after) {
        auto reshape_after_node = std::make_shared<ngraph::opset8::Reshape>(nodes.back(),
            std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::i64,
                ngraph::Shape{last_node_shape.size()}, last_node_shape), false);
        reshape_after_node->set_friendly_name(nodes.back()->get_friendly_name());
        ngraph::copy_runtime_info(nodes.back(), reshape_after_node);
        for (auto consumer : consumers) {
            consumer.replace_source_output(reshape_after_node);
        }
    }

    return need_reshape_before || need_reshape_after;
}

static std::shared_ptr<ngraph::Node> CreateMatmulPattern(
    std::shared_ptr<ngraph::Node>& input,
    std::shared_ptr<ngraph::Node>& matmul1,
    std::shared_ptr<ngraph::Node>& matmul2,
    const ngraph::pattern::op::ValuePredicate& pred = [](const ngraph::Output<ngraph::Node>& output) { return true; }) {
    auto constant = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto fake_quantize = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({constant,
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>()});
    auto matmul_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{constant, fake_quantize});
    input = ngraph::pattern::any_input([](const ngraph::Output<ngraph::Node>& node) {
        auto shape = node.get_node_shared_ptr()->get_output_shape(0);
        return shape.size() > 2 && std::count_if(shape.begin(), shape.end(), [](size_t e) { return e > 1; }) <= 2; });
    matmul1 = ngraph::pattern::wrap_type<ngraph::opset8::MatMul>({matmul_input, input}, pred);
    matmul2 = ngraph::pattern::wrap_type<ngraph::opset8::MatMul>({input, matmul_input}, pred);
    return std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{matmul1, matmul2});
}

InsertReshapeAroundMatmul::InsertReshapeAroundMatmul() {
    MATCHER_SCOPE(InsertReshapeAroundMatmul);

    auto pred = [](const ngraph::Output<ngraph::Node>& node) {
        const auto& outputs = node.get_node_shared_ptr()->outputs();
        const auto& inputs = outputs[0].get_target_inputs();
        if (inputs.empty()) {
            return true;
        }

        auto next_node = inputs.begin()->get_node();
        return outputs.size() != 1 ||
            !dynamic_cast<ngraph::opset8::Transpose*>(next_node) &&
            !dynamic_cast<ngraph::opset8::FakeQuantize*>(next_node) &&
            !dynamic_cast<ngraph::opset8::Add*>(next_node);
    };

    std::shared_ptr<ngraph::Node> input;
    std::shared_ptr<ngraph::Node> matmul1;
    std::shared_ptr<ngraph::Node> matmul2;
    auto matmul = CreateMatmulPattern(input, matmul1, matmul2, pred);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &matcher) {
        return InsertReshape(matcher, input, matmul1, matmul2);
    };

    auto matcher = std::make_shared<ngraph::pattern::Matcher>(matmul, "InsertReshapeAroundMatmul");
    this->register_matcher(matcher, callback);
}

InsertReshapeAroundMatmulWithAdd::InsertReshapeAroundMatmulWithAdd() {
    MATCHER_SCOPE(InsertReshapeAroundMatmulWithAdd);

    auto pred = [](const ngraph::Output<ngraph::Node>& node) {
        const auto& outputs = node.get_node_shared_ptr()->outputs();
        const auto& inputs = outputs[0].get_target_inputs();
        if (inputs.empty()) {
            return true;
        }

        auto next_node = inputs.begin()->get_node();
        return outputs.size() != 1 ||
            !dynamic_cast<ngraph::opset8::Transpose*>(next_node) &&
            !dynamic_cast<ngraph::opset8::FakeQuantize*>(next_node);
    };

    std::shared_ptr<ngraph::Node> input;
    std::shared_ptr<ngraph::Node> matmul1;
    std::shared_ptr<ngraph::Node> matmul2;
    auto matmul = CreateMatmulPattern(input, matmul1, matmul2);
    auto add_input = ngraph::pattern::any_input();
    auto add1 = ngraph::pattern::wrap_type<ngraph::opset8::Add>({matmul, add_input}, pred);
    auto add2 = ngraph::pattern::wrap_type<ngraph::opset8::Add>({add_input, matmul}, pred);
    auto add = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{add1, add2});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &matcher) {
        return InsertReshape(matcher, input, matmul1, matmul2, add1, add2);
    };

    auto matcher = std::make_shared<ngraph::pattern::Matcher>(add, "InsertReshapeAroundMatmulWithAdd");
    this->register_matcher(matcher, callback);
}

InsertReshapeAroundMatmulWithFq::InsertReshapeAroundMatmulWithFq() {
    MATCHER_SCOPE(InsertReshapeAroundMatmulWithFq);

    std::shared_ptr<ngraph::Node> input;
    std::shared_ptr<ngraph::Node> matmul1;
    std::shared_ptr<ngraph::Node> matmul2;
    auto matmul = CreateMatmulPattern(input, matmul1, matmul2);
    auto add_input = ngraph::pattern::any_input();
    auto add1 = ngraph::pattern::wrap_type<ngraph::opset8::Add>({matmul, add_input});
    auto add2 = ngraph::pattern::wrap_type<ngraph::opset8::Add>({add_input, matmul});
    auto fq_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{matmul, add1, add2});
    auto fake_quantize2 = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({fq_input, ngraph::pattern::any_input(),
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()},
        [](const ngraph::Output<ngraph::Node>& node) {
            const auto& outputs = node.get_node_shared_ptr()->outputs();
            const auto& inputs = outputs[0].get_target_inputs();
            if (inputs.empty()) {
                return true;
            }

            auto next_node = inputs.begin()->get_node();
            return outputs.size() != 1 ||
                !dynamic_cast<ngraph::opset8::Transpose*>(next_node);
        });

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &matcher) {
        return InsertReshape(matcher, input, matmul1, matmul2, add1, add2, fake_quantize2);
    };

    auto matcher = std::make_shared<ngraph::pattern::Matcher>(fake_quantize2, "InsertReshapeAroundMatmulWithFq");
    this->register_matcher(matcher, callback);
}

InsertReshapeAroundMatmulWithTranspose::InsertReshapeAroundMatmulWithTranspose() {
    MATCHER_SCOPE(InsertReshapeAroundMatmulWithTranspose);

    std::shared_ptr<ngraph::Node> input;
    std::shared_ptr<ngraph::Node> matmul1;
    std::shared_ptr<ngraph::Node> matmul2;
    auto matmul = CreateMatmulPattern(input, matmul1, matmul2);
    auto add_input = ngraph::pattern::any_input();
    auto add1 = ngraph::pattern::wrap_type<ngraph::opset8::Add>({matmul, add_input});
    auto add2 = ngraph::pattern::wrap_type<ngraph::opset8::Add>({add_input, matmul});
    auto fq_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{matmul, add1, add2});
    auto fake_quantize2 = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({fq_input, ngraph::pattern::any_input(),
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()});
    auto transpose_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{fq_input, fake_quantize2});
    auto transpose = ngraph::pattern::wrap_type<ngraph::opset8::Transpose>({transpose_input, ngraph::pattern::any_input()});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &matcher) {
        return InsertReshape(matcher, input, matmul1, matmul2, add1, add2, fake_quantize2, transpose);
    };

    auto matcher = std::make_shared<ngraph::pattern::Matcher>(transpose, "InsertReshapeAroundMatmulWithTranspose");
    this->register_matcher(matcher, callback);
}
} // namespace GNAPluginNS
