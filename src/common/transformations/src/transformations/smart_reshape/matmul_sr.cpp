// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/matmul_sr.hpp"

#include <memory>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/validation_util.hpp>
#include <numeric>

#include "itt.hpp"

namespace {

bool relax_hc_reshape_followed_by_matmul(const ngraph::pattern::PatternValueMap& pattern_to_output,
                                         const std::shared_ptr<ngraph::Node>& matmul_label,
                                         const std::shared_ptr<ngraph::Node>& reshape_label,
                                         const std::shared_ptr<ngraph::Node>& other_input_label,
                                         const std::shared_ptr<ngraph::Node>& reshape_pattern_label,
                                         bool reshape_is_A_input) {
    const auto& reshape_rank = pattern_to_output.at(reshape_label).get_partial_shape().rank();
    const auto& matmul =
        std::dynamic_pointer_cast<ngraph::opset4::MatMul>(pattern_to_output.at(matmul_label).get_node_shared_ptr());
    if (!matmul || reshape_rank.is_dynamic() || reshape_rank.get_length() != 2)
        return false;
    const auto& shape_source = pattern_to_output.at(other_input_label);
    if (ngraph::is_type<ngraph::opset4::Transpose>(shape_source.get_node_shared_ptr()) ||
        ngraph::is_type<ngraph::opset4::Reshape>(shape_source.get_node_shared_ptr()))
        // avoiding loop creation
        return false;

    const auto& raw_idx =
        reshape_is_A_input ? (matmul->get_transpose_b() ? -1 : -2) : (matmul->get_transpose_a() ? -2 : -1);
    const auto& idx = ngraph::normalize_axes(matmul->description(), {raw_idx}, reshape_rank);
    const auto& C = std::make_shared<ngraph::opset4::Gather>(
        std::make_shared<ngraph::opset4::ShapeOf>(shape_source),
        ngraph::opset4::Constant::create(ngraph::element::i64, {idx.size()}, idx),
        ngraph::opset4::Constant::create(ngraph::element::i64, {}, {0}));
    const auto& N = ngraph::opset4::Constant::create(ngraph::element::i64, {1}, {-1});
    const auto& pattern_vector =
        reshape_is_A_input ? (matmul->get_transpose_a() ? ngraph::OutputVector({C, N}) : ngraph::OutputVector({N, C}))
                           : (matmul->get_transpose_b() ? ngraph::OutputVector({N, C}) : ngraph::OutputVector({C, N}));
    const auto& new_reshape_pattern = std::make_shared<ngraph::opset4::Concat>(pattern_vector, 0);

    auto reshape_pattern = pattern_to_output.at(reshape_pattern_label).get_node_shared_ptr();
    new_reshape_pattern->set_friendly_name(reshape_pattern->get_friendly_name());
    copy_runtime_info(reshape_pattern, new_reshape_pattern);
    replace_node(reshape_pattern, new_reshape_pattern);
    return true;
}

}  // namespace

ngraph::pass::ReshapeAMatMul::ReshapeAMatMul() {
    // TODO: enable conditional compile
    // MATCHER_SCOPE(ReshapeAMatMul);
    auto other_input_label = pattern::any_input();
    auto reshape_input_label = pattern::any_input();
    auto reshape_pattern_label = pattern::any_input();
    auto reshape_label = ngraph::pattern::wrap_type<opset4::Reshape>({reshape_input_label, reshape_pattern_label});
    auto matmul_label = ngraph::pattern::wrap_type<opset4::MatMul>({reshape_label, other_input_label});

    matcher_pass_callback callback = [=](pattern::Matcher& m) -> bool {
        const auto& pattern_to_output = m.get_pattern_value_map();
        return relax_hc_reshape_followed_by_matmul(pattern_to_output,
                                                   matmul_label,
                                                   reshape_label,
                                                   other_input_label,
                                                   reshape_pattern_label,
                                                   true);
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(matmul_label /*, matcher_name */);
    register_matcher(m, callback);
}

ngraph::pass::ReshapeBMatMul::ReshapeBMatMul() {
    // TODO: enable conditional compile
    // MATCHER_SCOPE(ReshapeBMatMul);
    auto other_input_label = pattern::any_input();
    auto reshape_input_label = pattern::any_input();
    auto reshape_pattern_label = pattern::any_input();
    auto reshape_label = ngraph::pattern::wrap_type<opset4::Reshape>({reshape_input_label, reshape_pattern_label});
    auto matmul_label = ngraph::pattern::wrap_type<opset4::MatMul>({other_input_label, reshape_label});

    matcher_pass_callback callback = [=](pattern::Matcher& m) -> bool {
        const auto& pattern_to_output = m.get_pattern_value_map();
        return relax_hc_reshape_followed_by_matmul(pattern_to_output,
                                                   matmul_label,
                                                   reshape_label,
                                                   other_input_label,
                                                   reshape_pattern_label,
                                                   false);
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(matmul_label /*, matcher_name */);
    register_matcher(m, callback);
}

ngraph::pass::TransposeMatMul::TransposeMatMul() {
    // TODO: enable conditional compile
    // MATCHER_SCOPE(TransposeMatMul);
    auto matmul_label = ngraph::pattern::wrap_type<opset4::MatMul>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) -> bool {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto matmul =
            std::dynamic_pointer_cast<ngraph::opset4::MatMul>(pattern_to_output.at(matmul_label).get_node_shared_ptr());
        if (!matmul)
            return false;

        auto transpose_is_fusable = [](const std::shared_ptr<ngraph::Node>& input) {
            const auto& input_rank = input->get_output_partial_shape(0).rank();
            if (input_rank.is_static() && input_rank.get_length() >= 2) {
                if (auto transpose = std::dynamic_pointer_cast<ngraph::opset4::Transpose>(input)) {
                    if (auto order =
                            std::dynamic_pointer_cast<opset4::Constant>(transpose->get_input_node_shared_ptr(1))) {
                        const auto& order_vector = order->cast_vector<int64_t>();
                        std::vector<int64_t> fusable_order(input_rank.get_length());
                        std::iota(fusable_order.begin(), fusable_order.end(), 0);
                        std::swap(fusable_order[input_rank.get_length() - 1],
                                  fusable_order[input_rank.get_length() - 2]);
                        return order_vector == fusable_order;
                    }
                }
            }
            return false;
        };

        NodeVector fused_nodes;
        auto input_A = matmul->input_value(0);
        bool transpose_A = matmul->get_transpose_a();
        if (transpose_is_fusable(input_A.get_node_shared_ptr())) {
            fused_nodes.push_back(input_A.get_node_shared_ptr());
            input_A = input_A.get_node()->input_value(0);
            transpose_A = !transpose_A;
        }

        auto input_B = matmul->input_value(1);
        auto transpose_B = matmul->get_transpose_b();
        if (transpose_is_fusable(input_B.get_node_shared_ptr())) {
            fused_nodes.push_back(input_B.get_node_shared_ptr());
            input_B = input_B.get_node()->input_value(0);
            transpose_B = !transpose_B;
        }

        if (!fused_nodes.empty()) {
            auto updated_matmul = std::make_shared<opset4::MatMul>(input_A, input_B, transpose_A, transpose_B);
            fused_nodes.push_back(matmul);
            copy_runtime_info(fused_nodes, updated_matmul);
            updated_matmul->set_friendly_name(matmul->get_friendly_name());
            replace_node(matmul, updated_matmul);
            return true;
        }
        return false;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(matmul_label /*, matcher_name */);
    register_matcher(m, callback);
}
