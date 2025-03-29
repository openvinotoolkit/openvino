// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/matmul_sr.hpp"

#include <memory>
#include <numeric>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace {

bool relax_hc_reshape_followed_by_matmul(const ov::pass::pattern::PatternValueMap& pattern_to_output,
                                         const std::shared_ptr<ov::Node>& matmul_label,
                                         const std::shared_ptr<ov::Node>& reshape_label,
                                         const std::shared_ptr<ov::Node>& other_input_label,
                                         const std::shared_ptr<ov::Node>& reshape_pattern_label,
                                         bool reshape_is_A_input) {
    const auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(pattern_to_output.at(matmul_label).get_node_shared_ptr());
    if (!matmul)
        return false;
    const auto& shape_source = pattern_to_output.at(other_input_label);
    if (ov::is_type<ov::op::v1::Transpose>(shape_source.get_node_shared_ptr()) ||
        ov::is_type<ov::op::v1::Reshape>(shape_source.get_node_shared_ptr()))
        // avoiding loop creation
        return false;

    bool is_1d = ov::pass::pattern::rank_equals(1)(shape_source);
    int64_t idx = -1;
    if (!is_1d) {
        idx = reshape_is_A_input ? (matmul->get_transpose_b() ? -1 : -2) : (matmul->get_transpose_a() ? -2 : -1);
    }

    const auto in_C_0 = std::make_shared<ov::op::v3::ShapeOf>(shape_source);
    const auto in_C_1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {idx});
    const auto in_C_2 = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    const auto C = std::make_shared<ov::op::v8::Gather>(in_C_0, in_C_1, in_C_2);
    const auto N = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    const auto pattern_vector = reshape_is_A_input
                                    ? (matmul->get_transpose_a() ? ov::OutputVector({C, N}) : ov::OutputVector({N, C}))
                                    : (matmul->get_transpose_b() ? ov::OutputVector({N, C}) : ov::OutputVector({C, N}));
    const auto new_reshape_pattern = std::make_shared<ov::op::v0::Concat>(pattern_vector, 0);

    auto reshape_pattern = pattern_to_output.at(reshape_pattern_label).get_node_shared_ptr();
    ov::NodeVector nodes_to_copy_rt_info{new_reshape_pattern, C, N, in_C_0, in_C_1, in_C_2};
    copy_runtime_info(reshape_pattern, nodes_to_copy_rt_info);

    auto reshape_input = pattern_to_output.at(reshape_label).get_node_shared_ptr()->input(1);
    reshape_input.replace_source_output(new_reshape_pattern);
    return true;
}

}  // namespace

ov::pass::ReshapeAMatMul::ReshapeAMatMul() {
    MATCHER_SCOPE(ReshapeAMatMul);
    auto other_input_label = pattern::any_input(ov::pass::pattern::has_static_rank());
    auto reshape_input_label = pattern::any_input();
    auto reshape_pattern_label = pattern::any_input();
    auto reshape_predicate = [](ov::Output<ov::Node> output) -> bool {
        return ov::pass::pattern::rank_equals(2)(output) && ov::pass::pattern::consumers_count(1)(output);
    };
    auto reshape_label = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({reshape_input_label, reshape_pattern_label},
                                                                           reshape_predicate);
    auto matmul_label = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({reshape_label, other_input_label});

    matcher_pass_callback callback = [=](pattern::Matcher& m) -> bool {
        const auto& pattern_to_output = m.get_pattern_value_map();
        return relax_hc_reshape_followed_by_matmul(pattern_to_output,
                                                   matmul_label,
                                                   reshape_label,
                                                   other_input_label,
                                                   reshape_pattern_label,
                                                   true);
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(matmul_label, matcher_name);
    register_matcher(m, callback);
}

ov::pass::ReshapeBMatMul::ReshapeBMatMul() {
    MATCHER_SCOPE(ReshapeBMatMul);
    auto other_input_label = pattern::any_input(ov::pass::pattern::has_static_rank());
    auto reshape_input_label = pattern::any_input();
    auto reshape_pattern_label = pattern::any_input();
    auto reshape_predicate = [](ov::Output<ov::Node> output) -> bool {
        return ov::pass::pattern::rank_equals(2)(output) && ov::pass::pattern::consumers_count(1)(output);
    };
    auto reshape_label = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({reshape_input_label, reshape_pattern_label},
                                                                           reshape_predicate);
    auto matmul_label = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({other_input_label, reshape_label});

    matcher_pass_callback callback = [=](pattern::Matcher& m) -> bool {
        const auto& pattern_to_output = m.get_pattern_value_map();
        return relax_hc_reshape_followed_by_matmul(pattern_to_output,
                                                   matmul_label,
                                                   reshape_label,
                                                   other_input_label,
                                                   reshape_pattern_label,
                                                   false);
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(matmul_label, matcher_name);
    register_matcher(m, callback);
}

ov::pass::TransposeMatMul::TransposeMatMul() {
    MATCHER_SCOPE(TransposeMatMul);
    auto matmul_label = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) -> bool {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(pattern_to_output.at(matmul_label).get_node_shared_ptr());
        if (!matmul)
            return false;

        auto transpose_is_fusable = [](const std::shared_ptr<ov::Node>& input) {
            const auto& input_rank = input->get_output_partial_shape(0).rank();
            if (input_rank.is_static() && input_rank.get_length() >= 2) {
                if (auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(input)) {
                    if (auto order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->get_input_node_shared_ptr(1))) {
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
            auto updated_matmul = std::make_shared<ov::op::v0::MatMul>(input_A, input_B, transpose_A, transpose_B);
            fused_nodes.push_back(matmul);
            copy_runtime_info(fused_nodes, updated_matmul);
            updated_matmul->set_friendly_name(matmul->get_friendly_name());
            replace_node(matmul, updated_matmul);
            return true;
        }
        return false;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(matmul_label, matcher_name);
    register_matcher(m, callback);
}
