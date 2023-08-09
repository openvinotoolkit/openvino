// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/dereshape_matmul.hpp"

#include <openvino/core/dimension_tracker.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/util/binary_elementwise_arithmetic.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/symbolic_transformations/utils.hpp>

#include "itt.hpp"

ov::pass::DeReshapeMatMul::DeReshapeMatMul() {
    MATCHER_SCOPE(DeReshapeMatMul);

    auto reshape_0 = pattern::wrap_type<op::v1::Reshape>(pattern::has_static_rank());
    auto bea_0 = pattern::wrap_type<op::util::BinaryElementwiseArithmetic>({reshape_0, pattern::any_input()});
    auto or_0 = std::make_shared<pattern::op::Or>(OutputVector{reshape_0, bea_0});
    // FIXME: put all checks in the pattern of reshape and bea
    auto reshape_1 = pattern::wrap_type<op::v1::Reshape>(pattern::has_static_rank());
    auto bea_1 = pattern::wrap_type<op::util::BinaryElementwiseArithmetic>({reshape_1, pattern::any_input()});
    auto or_1 = std::make_shared<pattern::op::Or>(OutputVector{reshape_1, bea_1});
    // FIXME: put all checks in the pattern of reshape and bea
    auto matmul = pattern::wrap_type<op::v0::MatMul>({or_0, or_1});

    auto reshape_2 = pattern::wrap_type<op::v1::Reshape>({matmul, pattern::any_input()}, pattern::has_static_rank());

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        // find bottom Reshape check that output "batch" dims are equal to pre-first Reshape ones
        // and check that two last dims are same

        auto reshape_0_node = pattern_to_output.at(reshape_0).get_node_shared_ptr();
        auto reshape_1_node = pattern_to_output.at(reshape_1).get_node_shared_ptr();
        auto matmul_node = pattern_to_output.at(matmul).get_node_shared_ptr();
        if (reshape_0_node->get_input_partial_shape(0).rank().is_dynamic() ||
            reshape_0_node->get_output_partial_shape(0).rank().is_dynamic())
            return false;
        if (reshape_1_node->get_input_partial_shape(0).rank().is_dynamic() ||
            reshape_1_node->get_output_partial_shape(0).rank().is_dynamic())
            return false;
        if (reshape_0_node->get_input_partial_shape(0).size() != reshape_1_node->get_input_partial_shape(0).size())
            return false;
        if (reshape_0_node->get_output_partial_shape(0).size() != reshape_1_node->get_output_partial_shape(0).size())
            return false;
        if (!reshape_keeps_last_two_dims(reshape_0_node) || !reshape_keeps_last_two_dims(reshape_1_node))
            return false;
        if (!batches_are_equal(reshape_0_node, reshape_1_node))
            return false;
        // proved MatMul could have been executed on the non-Reshaped input tensors

        std::vector<Node*> nodes_for_revalidation{matmul_node.get()};
        Output<Node> output = matmul_node->output(0);
        // to reduce number of Reshapes -- searching for Reshape on the output of the MatMul skipping nodes which don't
        // influence output
        if (output.get_target_inputs().size() != 1)
            return false;
        auto reshape_output = ov::as_type<op::v1::Reshape>(output.get_target_inputs().begin()->get_node());
        if (!reshape_output)
            return false;  // we didn't find Reshape back on the output of the MatMul

        reshape_0_node->output(0).replace(reshape_0_node->input_value(0));
        reshape_1_node->output(0).replace(reshape_1_node->input_value(0));
        reshape_output->output(0).replace(reshape_output->input_value(0));
        for (auto& node : nodes_for_revalidation)
            node->validate_and_infer_types();
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(reshape_2, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::DeReshapeMatMulWithComplications::DeReshapeMatMulWithComplications() {
    MATCHER_SCOPE(DeReshapeMatMulWithComplications);

    // lhs of MatMul
    auto lhs_reshape =
        pattern::wrap_type<op::v1::Reshape>(pattern::op::as_value_predicate([](std::shared_ptr<Node> n) -> bool {
            return reshape_keeps_last_two_dims(n);
        }));

    // rhs of MatMul
    auto rhs_reshape =
        pattern::wrap_type<op::v1::Reshape>(pattern::op::as_value_predicate([](std::shared_ptr<Node> n) -> bool {
            return reshape_keeps_last_two_dims(n);
        }));
    auto rhs_concat =
        pattern::wrap_type<op::v0::Concat>({pattern::any_input(), rhs_reshape},
                                           pattern::op::as_value_predicate([](std::shared_ptr<Node> n) -> bool {
                                               auto output_pshape = n->get_output_partial_shape(0);
                                               if (output_pshape.rank().is_dynamic() || output_pshape.size() <= 2)
                                                   return false;
                                               const auto& concat = ov::as_type_ptr<ov::op::v0::Concat>(n);
                                               if (!concat)
                                                   return false;
                                               return concat->get_concatenation_axis() >= output_pshape.size() - 2;
                                           }));
    auto rhs_reshape_or_concat = std::make_shared<pattern::op::Or>(OutputVector{rhs_reshape, rhs_concat});

    auto rhs_bea_scalar = pattern::any_input([](ov::Output<Node> out) {
        return out.get_partial_shape().is_static() && ov::shape_size(out.get_shape()) == 1;
    });
    auto rhs_bea = pattern::wrap_type<op::util::BinaryElementwiseArithmetic>({rhs_reshape_or_concat, rhs_bea_scalar});
    auto rhs_bea_or_concat = std::make_shared<pattern::op::Or>(OutputVector{rhs_reshape_or_concat, rhs_bea});

    auto matmul = pattern::wrap_type<op::v0::MatMul>({lhs_reshape, rhs_bea_or_concat});

    auto constant = pattern::wrap_type<op::v0::Constant>();
    auto zero_dim_value = pattern::wrap_type<op::v1::Multiply>({constant, pattern::any_input()});
    auto dimensions_concat =
        pattern::wrap_type<op::v0::Concat>({zero_dim_value, pattern::any_input(), pattern::any_input()});
    auto add_reshape = pattern::wrap_type<op::v1::Reshape>({pattern::any_input(), dimensions_concat});
    auto add = pattern::wrap_type<op::v1::Add>({matmul, add_reshape});

    auto matmul_or_add = std::make_shared<pattern::op::Or>(OutputVector{matmul, add});
    auto final_reshape = pattern::wrap_type<op::v1::Reshape>({matmul_or_add, pattern::any_input()});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_node = m.get_pattern_map();
        const auto& pattern_to_output = m.get_pattern_value_map();
        std::vector<Node*> nodes_for_revalidation{pattern_to_node.at(matmul).get()};

        // reshapes check: BEGIN
        // reshape_keeps_last_two_dims checks were already done in the pattern
        auto reshape_0_node = pattern_to_node.at(lhs_reshape);
        auto reshape_1_node = pattern_to_node.at(rhs_reshape);
        if (!batches_are_equal(reshape_0_node, reshape_1_node))
            return false;
        auto reshape_2_node = pattern_to_node.at(final_reshape);
        if (!batches_are_equal(reshape_0_node->get_input_partial_shape(0),
                               reshape_2_node->get_output_partial_shape(0),
                               true) &&
            !batches_are_equal(reshape_1_node->get_input_partial_shape(0),
                               reshape_2_node->get_output_partial_shape(0),
                               true))
            return false;
        // reshapes check: END

        // checks for Add -- if we can optimally delete Reshapes
        if (pattern_to_node.count(add) && pattern_to_node.count(add_reshape)) {
            auto add_reshape_node = pattern_to_node.at(add_reshape);
            auto output_rank = add_reshape_node->get_output_partial_shape(0).rank();
            auto reshape_rank = reshape_0_node->get_input_partial_shape(0).rank();
            auto input_shape = reshape_0_node->get_input_partial_shape(0);
            auto pattern_concat = add_reshape_node->get_input_node_shared_ptr(1);
            if (output_rank.is_dynamic() || output_rank != 3 || reshape_rank != 4 ||
                !as_type_ptr<op::v0::Concat>(pattern_concat) || input_shape.rank().is_dynamic() ||
                input_shape.rank() != 4)
                return false;

            int64_t constant_dim_of_batch = -1;
            int64_t constant_dim_idx = -1;
            if (input_shape[0].is_static()) {
                constant_dim_of_batch = input_shape[0].get_length();
                constant_dim_idx = 0;
            } else if (input_shape[1].is_static()) {
                constant_dim_of_batch = input_shape[1].get_length();
                constant_dim_idx = 1;
            }
            if (constant_dim_of_batch == -1 || constant_dim_idx == -1)
                return false;
            if (pattern_concat->input_value(0).get_shape() != Shape{1})
                return false;
            if (!as_type_ptr<op::v1::Multiply>(pattern_concat->get_input_node_shared_ptr(0)))
                return false;
            int64_t const_val = -1;
            if (auto constant = ov::as_type_ptr<op::v0::Constant>(
                    pattern_concat->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(0))) {
                const_val = constant->cast_vector<int64_t>()[0];
            } else if (auto constant = ov::as_type_ptr<op::v0::Constant>(
                           pattern_concat->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(1))) {
                const_val = constant->cast_vector<int64_t>()[0];
            }
            if (const_val == -1 || constant_dim_of_batch != const_val)
                return false;

            auto mm_output_shape = pattern_to_output.at(matmul).get_partial_shape();
            auto final_reshape_shape = pattern_to_output.at(final_reshape).get_partial_shape();
            if (!last_two_dims_are_equal(mm_output_shape, final_reshape_shape))
                return false;

            const auto& add_output_shape = pattern_to_node.at(add)->get_output_partial_shape(0);
            auto add_dim = add_output_shape[output_rank.get_length() - 1];
            auto table = ov::DimensionTracker::get_table_of_equivalence(add_dim);
            if (!table)
                return false;
            table->set_as_equal(add_dim, mm_output_shape[output_rank.get_length() - 1]);
            if (!reshape_keeps_last_two_dims(reshape_2_node))
                return false;

            auto concat_inputs = pattern_concat->input_values();
            std::vector<int64_t> new_batch_constant_val = {(constant_dim_idx == 0 ? constant_dim_of_batch : -1),
                                                           (constant_dim_idx == 0 ? -1 : constant_dim_of_batch)};
            concat_inputs[0] =
                op::v0::Constant::create(concat_inputs[0].get_element_type(), Shape{2}, new_batch_constant_val);
            auto new_shape_concat = std::make_shared<op::v0::Concat>(concat_inputs, 0);
            // TODO: take cate of -1s and 0s in the input of this reshape
            auto new_add_reshape = std::make_shared<op::v1::Reshape>(add_reshape_node, new_shape_concat, false);
            pattern_to_node.at(add)
                ->input(!bool(pattern_to_node.at(matmul)->output(0).get_target_inputs().begin()->get_index()))
                .replace_source_output(new_add_reshape);
            nodes_for_revalidation.push_back(pattern_to_node.at(add).get());
        } else {
            if (!reshape_keeps_last_two_dims(reshape_2_node))
                return false;
        }
        // resolving Reshape on the rhs branch with Concat
        if (auto concat_node = ov::as_type_ptr<op::v0::Concat>(pattern_to_node.at(rhs_concat))) {
            auto axis = concat_node->get_concatenation_axis();
            if (axis == concat_node->get_output_partial_shape(0).size() - 1)
                axis = -1;
            else
                axis = -2;
            auto target_shape_of_input =
                get_shape_from_sources(reshape_1_node->input_value(0), concat_node->input_value(0));
            auto input_reshape =
                std::make_shared<op::v1::Reshape>(concat_node->input_value(0), target_shape_of_input, false);
            reshape_1_node->output(0).replace(reshape_1_node->input_value(0));
            auto new_concat =
                concat_node->clone_with_new_inputs({input_reshape->output(0), concat_node->input_value(1)});
            ov::as_type_ptr<ov::op::v0::Concat>(new_concat)->set_axis(axis);
            ov::as_type_ptr<ov::op::v0::Concat>(new_concat)->set_concatenation_axis(axis);
            new_concat->validate_and_infer_types();
            auto target_shape_of_output = get_shape_from_sources(input_reshape->input_value(0), new_concat->output(0));
            auto output_reshape =
                std::make_shared<op::v1::Reshape>(new_concat->output(0), target_shape_of_output, false);
            if (pattern_to_node.count(rhs_bea)) {
                auto bea = pattern_to_node.at(rhs_bea);
                auto idx_of_non_scalar_data = bea->input_value(0) == pattern_to_output.at(rhs_bea_scalar) ? 1 : 0;
                bea->input(idx_of_non_scalar_data).replace_source_output(new_concat);
                nodes_for_revalidation.push_back(bea.get());
            } else {
                auto rhs_idx = 1;
                pattern_to_node.at(matmul)->input(rhs_idx).replace_source_output(new_concat);
            }
            concat_node->output(0).replace(output_reshape->output(0));
            output_reshape->set_friendly_name(concat_node->get_friendly_name());
        }

        // resolving Reshape on the lhs branch
        reshape_0_node->output(0).replace(reshape_0_node->input_value(0));

        reshape_2_node->output(0).replace(reshape_2_node->input_value(0));

        for (auto& node : nodes_for_revalidation)
            node->validate_and_infer_types();
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(final_reshape, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
