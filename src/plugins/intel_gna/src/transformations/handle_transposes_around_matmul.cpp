// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/handle_transposes_around_matmul.hpp"

#include <ie/ie_common.h>

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <numeric>
#include <openvino/cc/ngraph/itt.hpp>

#include "backend/gna_limitations.hpp"
#include "common/graph_utils.hpp"

using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::limitations;
using namespace ov::intel_gna::graph_utils;

namespace {

void ReplaceTransposeWithReshape(std::shared_ptr<ngraph::Node> transpose_node) {
    auto shape = transpose_node->get_output_shape(0);
    auto reshape_const =
        std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{shape.size()}, shape);
    auto reshape_node = std::make_shared<ngraph::opset8::Reshape>(transpose_node->input_value(0), reshape_const, false);
    reshape_node->set_friendly_name(transpose_node->get_friendly_name());
    ngraph::copy_runtime_info(transpose_node, {reshape_node, reshape_const});
    transpose_node->output(0).replace(reshape_node->output(0));
}

void InsertTranspose(std::shared_ptr<ngraph::Node> prev_node, const std::string& base_name, bool before_matmul) {
    ngraph::NodeVector new_ops;
    auto create_reshape =
        [&new_ops](const ngraph::Shape& shape, std::shared_ptr<ngraph::Node> input_node, const std::string& name) {
            auto reshape_const = std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::i64,
                                                                            ngraph::Shape{shape.size()},
                                                                            shape);
            new_ops.push_back(reshape_const);
            auto node = std::make_shared<ngraph::opset8::Reshape>(input_node, reshape_const, false);
            new_ops.push_back(node);
            node->set_friendly_name(name);
            return node;
        };

    auto consumers = prev_node->output(0).get_target_inputs();
    const auto orig_shape = prev_node->get_output_shape(0);
    std::vector<size_t> transpose_ids;
    for (size_t i = 0; i < orig_shape.size(); ++i) {
        if (orig_shape[i] > 1) {
            transpose_ids.push_back(i);
        }
    }
    IE_ASSERT(transpose_ids.size() == 2);
    std::vector<size_t> permute_order(orig_shape.size());
    std::iota(std::begin(permute_order), std::end(permute_order), 0);
    std::swap(permute_order[transpose_ids[0]], permute_order[transpose_ids[1]]);

    std::shared_ptr<ngraph::Node> node = prev_node;
    if (!before_matmul) {
        auto shape = prev_node->get_output_shape(0);
        std::swap(shape[0], shape[1]);
        node = create_reshape(shape, node, base_name + "/reshape_before_transpose");
    }

    auto transpose_order =
        ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{permute_order.size()}, permute_order);
    new_ops.push_back(transpose_order);
    node = std::make_shared<ngraph::opset8::Transpose>(node, transpose_order);
    node->set_friendly_name(base_name + "/in_transpose");
    new_ops.push_back(node);

    if (before_matmul) {
        node = create_reshape(orig_shape, node, base_name + "/reshape_after_transpose");
    }

    ngraph::copy_runtime_info(prev_node, new_ops);

    for (auto& input : consumers) {
        input.replace_source_output(node);
    }
}

bool VerifyReshape(const ngraph::Output<ngraph::Node>& reshape_out) {
    auto in_shape = reshape_out.get_node_shared_ptr()->get_input_shape(0);
    auto out_shape = reshape_out.get_node_shared_ptr()->get_output_shape(0);

    const auto is_input_scalar = in_shape.empty();
    const auto is_output_scalar = out_shape.empty();

    if (is_input_scalar && is_output_scalar) {
        // If both are scalar it means we don't need reshape.
        return false;
    }

    if (is_input_scalar || is_output_scalar) {
        // If one is scalar it means we need reshape.
        return true;
    }

    return in_shape[0] != out_shape[0];
}

bool VerifyConcat(const ngraph::Output<ngraph::Node>& node) {
    auto concat_node = std::dynamic_pointer_cast<ngraph::opset8::Concat>(node.get_node_shared_ptr());
    return concat_node && (concat_node->get_axis() == 0);
}

}  // namespace

HandleTransposeBeforeMatMul::HandleTransposeBeforeMatMul() {
    auto concat1 = ngraph::pattern::wrap_type<ngraph::opset8::Concat>(VerifyConcat);
    auto reshape1 = ngraph::pattern::wrap_type<ngraph::opset8::Reshape>(VerifyReshape);
    auto transpose_input1 = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{concat1, reshape1});
    auto transpose1 =
        ngraph::pattern::wrap_type<ngraph::opset8::Transpose>({transpose_input1, ngraph::pattern::any_input()});

    auto concat2 = ngraph::pattern::wrap_type<ngraph::opset8::Concat>(VerifyConcat);
    auto reshape2 = ngraph::pattern::wrap_type<ngraph::opset8::Reshape>(VerifyReshape);
    auto transpose_input2 = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{concat2, reshape2});
    auto transpose2 =
        ngraph::pattern::wrap_type<ngraph::opset8::Transpose>({transpose_input2, ngraph::pattern::any_input()});

    auto constant = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto fq = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({constant,
                                                                        ngraph::pattern::any_input(),
                                                                        ngraph::pattern::any_input(),
                                                                        ngraph::pattern::any_input(),
                                                                        ngraph::pattern::any_input()});

    auto matmul1 = ngraph::pattern::wrap_type<ngraph::opset8::MatMul>(
        {std::make_shared<ngraph::pattern::op::Or>(
             ngraph::OutputVector{reshape1, concat1, transpose1, constant, fq, ngraph::pattern::any_input()}),
         std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{reshape2, concat2, transpose2})});

    auto matmul2 = ngraph::pattern::wrap_type<ngraph::opset8::MatMul>(
        {std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{reshape1, concat1, transpose1, constant, fq}),
         ngraph::pattern::any_input()});

    auto matmul = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{matmul1, matmul2});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& matcher) {
        const auto& pattern_map = matcher.get_pattern_value_map();
        auto matmul_iter = pattern_map.find(matmul1);
        if (matmul_iter == std::end(pattern_map) &&
            (matmul_iter = pattern_map.find(matmul2)) == std::end(pattern_map)) {
            return false;
        }

        auto matmul_node = matmul_iter->second.get_node_shared_ptr();
        auto transpose_reshape_it = pattern_map.find(transpose1);
        if (transpose_reshape_it != std::end(pattern_map)) {
            ReplaceTransposeWithReshape(transpose_reshape_it->second.get_node_shared_ptr());
        } else {
            std::shared_ptr<ngraph::Node> prev_node = nullptr;
            if ((transpose_reshape_it = pattern_map.find(reshape1)) != std::end(pattern_map)) {
                prev_node = pattern_map.at(reshape1).get_node_shared_ptr();
            } else if ((transpose_reshape_it = pattern_map.find(concat1)) != std::end(pattern_map)) {
                prev_node = pattern_map.at(concat1).get_node_shared_ptr();
            }

            if (prev_node) {
                if (graph_utils::is_shape_2d(prev_node->get_output_shape(0)) &&
                    Limitations::is_transpose_supported(prev_node->get_output_shape(0))) {
                    InsertTranspose(prev_node, matmul_node->get_friendly_name(), true);
                }
            }
        }

        // Transpose the first input if it's a constant
        auto iter = pattern_map.find(fq);
        if (iter != pattern_map.end() || (iter = pattern_map.find(constant)) != pattern_map.end()) {
            auto prev_node = iter->second.get_node_shared_ptr();
            if (is_shape_2d(prev_node->get_output_shape(0))) {
                InsertTranspose(prev_node, prev_node->get_friendly_name(), true);
            }
        }

        transpose_reshape_it = pattern_map.find(transpose2);
        if (transpose_reshape_it != std::end(pattern_map)) {
            ReplaceTransposeWithReshape(transpose_reshape_it->second.get_node_shared_ptr());
        } else {
            std::shared_ptr<ngraph::Node> prev_node = nullptr;
            if ((transpose_reshape_it = pattern_map.find(reshape2)) != std::end(pattern_map)) {
                prev_node = pattern_map.at(reshape2).get_node_shared_ptr();
            } else if ((transpose_reshape_it = pattern_map.find(concat2)) != std::end(pattern_map)) {
                prev_node = pattern_map.at(concat2).get_node_shared_ptr();
            }

            if (prev_node) {
                if (graph_utils::is_shape_2d(prev_node->get_output_shape(0)) &&
                    Limitations::is_transpose_supported(prev_node->get_output_shape(0))) {
                    InsertTranspose(prev_node, matmul_node->get_friendly_name(), true);
                }
            }
        }
        return true;
    };

    auto matcher = std::make_shared<ngraph::pattern::Matcher>(matmul, "HandleTransposeBeforeMatMul");
    this->register_matcher(matcher, callback);
}

HandleTransposeAfterMatMul::HandleTransposeAfterMatMul() {
    auto matmul = ngraph::pattern::wrap_type<ngraph::opset8::MatMul>({}, [](const ngraph::Output<ngraph::Node>& node) {
        auto out_shape = node.get_node_shared_ptr()->get_output_shape(0);
        return std::count_if(out_shape.begin(), out_shape.end(), [](size_t n) {
                   return n > 1;
               }) > 1;
    });
    auto fq1 = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({matmul,
                                                                         ngraph::pattern::any_input(),
                                                                         ngraph::pattern::any_input(),
                                                                         ngraph::pattern::any_input(),
                                                                         ngraph::pattern::any_input()});
    auto add_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{matmul, fq1});
    auto add_left = ngraph::pattern::wrap_type<ngraph::opset8::Add>({add_input, ngraph::pattern::any_input()});
    auto add_right = ngraph::pattern::wrap_type<ngraph::opset8::Add>({ngraph::pattern::any_input(), add_input});
    auto fq2_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{matmul, add_left, add_right});
    auto fq2 = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({fq2_input,
                                                                         ngraph::pattern::any_input(),
                                                                         ngraph::pattern::any_input(),
                                                                         ngraph::pattern::any_input(),
                                                                         ngraph::pattern::any_input()});
    auto act_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{fq2_input, fq2});
    auto act = ngraph::pattern::wrap_type<ngraph::opset8::Relu,
                                          ngraph::opset8::Sigmoid,
                                          ngraph::opset8::Tanh,
                                          ngraph::opset8::Abs,
                                          ngraph::opset8::Log,
                                          ngraph::opset8::Exp,
                                          ngraph::opset8::Sign,
                                          ngraph::opset8::Clamp>({act_input});
    auto transpose_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{act_input, act});
    auto transpose =
        ngraph::pattern::wrap_type<ngraph::opset8::Transpose>({transpose_input, ngraph::pattern::any_input()});
    auto reshape_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{transpose_input, transpose});
    auto reshape = ngraph::pattern::wrap_type<ngraph::opset8::Reshape>({reshape_input, ngraph::pattern::any_input()},
                                                                       VerifyReshape);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& matcher) {
        const auto& pattern_map = matcher.get_pattern_value_map();
        auto transpose_it = pattern_map.find(transpose);
        if (transpose_it != std::end(pattern_map)) {
            ReplaceTransposeWithReshape(transpose_it->second.get_node_shared_ptr());
        } else {
            auto reshape_node = pattern_map.at(reshape).get_node_shared_ptr();
            if (!Limitations::is_transpose_supported(reshape_node->get_input_shape(0)))
                return false;
            auto iter = pattern_map.find(act);
            if (iter == pattern_map.end() && (iter = pattern_map.find(fq2)) == pattern_map.end() &&
                (iter = pattern_map.find(add_left)) == pattern_map.end() &&
                (iter = pattern_map.find(add_right)) == pattern_map.end() &&
                (iter = pattern_map.find(matmul)) == pattern_map.end()) {
                return false;
            }
            auto node = iter->second.get_node_shared_ptr();
            InsertTranspose(node, node->get_friendly_name(), false);
        }
        return true;
    };

    auto matcher = std::make_shared<ngraph::pattern::Matcher>(reshape, "HandleTransposeAfterMatMul");
    this->register_matcher(matcher, callback);
}

HandleTransposesAroundMatMul::HandleTransposesAroundMatMul() {
    add_matcher<HandleTransposeBeforeMatMul>();
    add_matcher<HandleTransposeAfterMatMul>();
}
