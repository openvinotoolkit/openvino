// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/handle_transposes_around_matmul.hpp"

#include <numeric>

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/rt_info.hpp>

#include "gna_plugin_log.hpp"
#include "backend/gna_limitations.hpp"

using namespace GNAPluginNS;

NGRAPH_RTTI_DEFINITION(HandleTransposesAroundMatMul, "HandleTransposesAroundMatMul", 0);
NGRAPH_RTTI_DEFINITION(HandleTransposeBeforeMatMul, "HandleTransposeBeforeMatMul", 0);
NGRAPH_RTTI_DEFINITION(HandleTransposeAfterMatMul, "HandleTransposeAfterMatMul", 0);

static void ReplaceTransposeWithReshape(std::shared_ptr<ngraph::Node> transpose_node) {
    auto shape = transpose_node->get_output_shape(0);
    auto reshape_const = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
        ngraph::Shape{shape.size()}, shape);
    auto reshape_node = std::make_shared<ngraph::opset7::Reshape>(transpose_node->input_value(0), reshape_const, false);
    reshape_node->set_friendly_name(transpose_node->get_friendly_name() + "/reshape");
    ngraph::copy_runtime_info(transpose_node, reshape_node);
    transpose_node->output(0).replace(reshape_node->output(0));
}

static void InsertTranspose(std::shared_ptr<ngraph::Node> prev_node, const std::string& base_name) {
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

    auto transpose_order = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{permute_order.size()}, permute_order);
    auto transpose = std::make_shared<ngraph::opset7::Transpose>(prev_node, transpose_order);
    transpose->set_friendly_name(base_name + "/in_transpose");

    auto reshapeConstAfter = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
        ngraph::Shape{orig_shape.size()}, orig_shape);
    auto reshapeAfter = std::make_shared<ngraph::opset7::Reshape>(transpose, reshapeConstAfter, false);
    reshapeAfter->set_friendly_name(base_name + "/reshape_after_transpose");
    ngraph::copy_runtime_info(prev_node, ngraph::NodeVector{transpose, reshapeAfter});

    for (auto input : consumers) {
        input.replace_source_output(reshapeAfter);
    }
}

HandleTransposeBeforeMatMul::HandleTransposeBeforeMatMul() {
    auto reshape = ngraph::pattern::wrap_type<ngraph::opset7::Reshape>({ngraph::pattern::any_input(),
        ngraph::pattern::any_input()}, VerifyReshape());
    auto transpose = ngraph::pattern::wrap_type<ngraph::opset7::Transpose>({reshape,
        ngraph::pattern::any_input()});
    auto matmul_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{reshape, transpose});
    auto matmul1 = ngraph::pattern::wrap_type<ngraph::opset7::MatMul>({matmul_input, ngraph::pattern::any_input()});
    auto matmul2 = ngraph::pattern::wrap_type<ngraph::opset7::MatMul>({ngraph::pattern::any_input(), matmul_input});
    auto matmul = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{matmul1, matmul2});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto transpose_it = pattern_map.find(transpose);
        if (transpose_it != std::end(pattern_map)) {
            ReplaceTransposeWithReshape(transpose_it->second.get_node_shared_ptr());
        } else {
            auto reshape_node = pattern_map.at(reshape).get_node_shared_ptr();
            if (!GNALimitations::IsTransposeSupported(reshape_node->get_output_shape(0))) return false;
            auto matmul_it = pattern_map.find(matmul1);
            auto matmul_out = matmul_it != std::end(pattern_map) ? matmul_it->second : pattern_map.at(matmul2);
            InsertTranspose(reshape_node, matmul_out.get_node_shared_ptr()->get_friendly_name());
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matmul, "HandleTransposeBeforeMatMul");
    this->register_matcher(m, callback);
}

HandleTransposeAfterMatMul::HandleTransposeAfterMatMul() {
    auto matmul = ngraph::pattern::wrap_type<ngraph::opset7::MatMul>();
    auto fq = ngraph::pattern::wrap_type<ngraph::opset7::FakeQuantize>({matmul, ngraph::pattern::any_input(),
        ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()});
    auto transpose_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{matmul, fq});
    auto transpose = ngraph::pattern::wrap_type<ngraph::opset7::Transpose>({transpose_input, ngraph::pattern::any_input()});
    auto reshape_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{transpose_input, transpose});
    auto reshape = ngraph::pattern::wrap_type<ngraph::opset7::Reshape>({reshape_input,
        ngraph::pattern::any_input()}, VerifyReshape());

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto transpose_it = pattern_map.find(transpose);
        if (transpose_it != std::end(pattern_map)) {
            ReplaceTransposeWithReshape(transpose_it->second.get_node_shared_ptr());
        } else {
            auto reshape_node = pattern_map.at(reshape).get_node_shared_ptr();
            if (!GNALimitations::IsTransposeSupported(reshape_node->get_input_shape(0))) return false;
            auto matmul_node = pattern_map.at(matmul).get_node_shared_ptr();
            InsertTranspose(matmul_node, matmul_node->get_friendly_name());
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape, "HandleTransposeAfterMatMul");
    this->register_matcher(m, callback);
}

bool VerifyReshape::operator()(const ngraph::Output<ngraph::Node>& reshape_out) const {
    auto in_shape = reshape_out.get_node_shared_ptr()->get_input_shape(0);
    auto out_shape = reshape_out.get_node_shared_ptr()->get_output_shape(0);

    // Check if Reshape changes the final 2d shape of Affine primitive
    in_shape.erase(std::remove(in_shape.begin(), in_shape.end(), 1), in_shape.end());
    out_shape.erase(std::remove(out_shape.begin(), out_shape.end(), 1), out_shape.end());
    return in_shape != out_shape;
}

HandleTransposesAroundMatMul::HandleTransposesAroundMatMul() {
    add_matcher<HandleTransposeBeforeMatMul>();
    add_matcher<HandleTransposeAfterMatMul>();
}
