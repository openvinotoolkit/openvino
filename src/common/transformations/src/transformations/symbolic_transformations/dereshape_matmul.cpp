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
#include "transformations/utils/utils.hpp"

bool concat_predicate(ov::Output<ov::Node> output) {
    auto output_pshape = output.get_partial_shape();
    if (output_pshape.rank().is_dynamic() || output_pshape.size() <= 2)
        return false;
    const auto& concat = ov::as_type_ptr<ov::op::v0::Concat>(output.get_node_shared_ptr());
    if (!concat)
        return false;
    return concat->get_concatenation_axis() >= output_pshape.rank().get_length() - 2;
}

ov::Output<ov::Node> get_shape_from_sources(const ov::Output<ov::Node>& batch_dims_source,
                                            const ov::Output<ov::Node>& non_batch_dims_source,
                                            const std::vector<std::shared_ptr<ov::Node>>& copy_rt_info_from) {
    auto batch_indices = std::vector<size_t>(batch_dims_source.get_partial_shape().size() - 2);
    std::iota(batch_indices.begin(), batch_indices.end(), 0);
    auto batch_dims = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(batch_dims_source,
                                                                                         batch_indices,
                                                                                         copy_rt_info_from);
    auto non_batch_indices = std::vector<size_t>(2);
    std::iota(non_batch_indices.begin(), non_batch_indices.end(), non_batch_dims_source.get_partial_shape().size() - 2);
    auto non_batch_dims = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(non_batch_dims_source,
                                                                                             non_batch_indices,
                                                                                             copy_rt_info_from);
    auto target_shape =
        ov::op::util::make_try_fold<ov::op::v0::Concat>(ov::OutputVector{batch_dims, non_batch_dims}, 0);
    ov::copy_runtime_info(copy_rt_info_from, target_shape);
    return target_shape->output(0);
}

#define IN_RESHAPE                                                                                            \
    pattern::wrap_type<op::v1::Reshape>(pattern::op::as_value_predicate([](std::shared_ptr<Node> n) -> bool { \
        return pattern::consumers_count(1)(n->output(0)) && reshape_keeps_last_two_dims(n);                   \
    }));

#define SCALAR_INPUT                                                                        \
    pattern::any_input([](ov::Output<Node> out) {                                           \
        return out.get_partial_shape().is_static() && ov::shape_size(out.get_shape()) == 1; \
    });

void pull_reshape_through_optional_concat_and_bea(const ov::pass::pattern::PatternValueMap& vm,
                                                  std::shared_ptr<ov::Node> concat_label,
                                                  std::shared_ptr<ov::Node> bea_label,
                                                  ov::Output<ov::Node> reshape_output,
                                                  ov::Input<ov::Node> matmul_input,
                                                  std::vector<ov::Node*>& nodes_for_revalidation) {
    // Reshape -- [Concat] -- [BEA with scalar] -- > MatMul
    auto original_reshape = reshape_output.get_node_shared_ptr();
    if (vm.count(concat_label)) {
        auto concat_node = ov::as_type_ptr<ov::op::v0::Concat>(vm.at(concat_label).get_node_shared_ptr());
        OPENVINO_ASSERT(concat_node != nullptr,
                        "DeReshapeMatMul transformation matched operation which should be Concat -- but it is not");
        auto rank = concat_node->get_output_partial_shape(0).rank().get_length();
        auto axis = (concat_node->get_concatenation_axis() == (rank - 1)) ? -1 : -2;

        auto idx_of_reshape_input = reshape_output == concat_node->input_value(0) ? 0 : 1;
        auto idx_of_non_reshape_input = static_cast<size_t>(!idx_of_reshape_input);

        auto target_shape_of_input = get_shape_from_sources(original_reshape->input_value(0),
                                                            concat_node->input_value(idx_of_non_reshape_input),
                                                            {original_reshape});

        auto input_reshape = original_reshape->clone_with_new_inputs(
            {concat_node->input_value(idx_of_non_reshape_input), target_shape_of_input});
        ov::copy_runtime_info(original_reshape, input_reshape);

        ov::replace_output_update_name(reshape_output, original_reshape->input_value(0));

        ov::OutputVector new_concat_inputs(2);
        new_concat_inputs[idx_of_reshape_input] = concat_node->input_value(idx_of_reshape_input);
        new_concat_inputs[idx_of_non_reshape_input] = input_reshape->output(0);

        auto new_concat = std::make_shared<ov::op::v0::Concat>(new_concat_inputs, axis);
        ov::copy_runtime_info({concat_node, original_reshape}, new_concat);

        auto target_shape_of_output =
            get_shape_from_sources(input_reshape->input_value(0), new_concat->output(0), {original_reshape});
        auto output_reshape = original_reshape->clone_with_new_inputs({new_concat->output(0), target_shape_of_output});
        ov::copy_runtime_info(original_reshape, output_reshape);

        if (vm.count(bea_label)) {
            auto bea_node = vm.at(bea_label).get_node_shared_ptr();
            auto idx_of_non_scalar_data = bea_node->input_value(0) == vm.at(concat_label) ? 0 : 1;
            bea_node->input(idx_of_non_scalar_data).replace_source_output(new_concat);
            nodes_for_revalidation.push_back(bea_node.get());
        } else {
            matmul_input.replace_source_output(new_concat);
        }
        ov::replace_output_update_name(concat_node->output(0), output_reshape->output(0));
    } else {
        // no Concat and it doesn't matter if BEA is present -- just delete reshape
        ov::replace_output_update_name(reshape_output, original_reshape->input_value(0));
    }
}

ov::pass::DeReshapeMatMul::DeReshapeMatMul() {
    MATCHER_SCOPE(DeReshapeMatMul);
    // BEGIN: symmetrical patterns for MatMul inputs

    // lhs of MatMul
    auto lhs_reshape = IN_RESHAPE;

    auto lhs_concat_0 = pattern::wrap_type<op::v0::Concat>({pattern::any_input(), lhs_reshape}, concat_predicate);
    auto lhs_concat_1 = pattern::wrap_type<op::v0::Concat>({lhs_reshape, pattern::any_input()}, concat_predicate);
    auto lhs_concat = std::make_shared<pattern::op::Or>(OutputVector{lhs_concat_0, lhs_concat_1});

    auto lhs_reshape_or_concat = std::make_shared<pattern::op::Or>(OutputVector{lhs_reshape, lhs_concat});

    auto lhs_bea_scalar = SCALAR_INPUT;
    auto lhs_bea = pattern::wrap_type<op::util::BinaryElementwiseArithmetic>({lhs_reshape_or_concat, lhs_bea_scalar},
                                                                             pattern::consumers_count(1));

    auto lhs_bea_or_concat = std::make_shared<pattern::op::Or>(OutputVector{lhs_reshape_or_concat, lhs_bea});

    // rhs of MatMul
    auto rhs_reshape = IN_RESHAPE;

    auto rhs_concat_0 = pattern::wrap_type<op::v0::Concat>({pattern::any_input(), rhs_reshape}, concat_predicate);
    auto rhs_concat_1 = pattern::wrap_type<op::v0::Concat>({rhs_reshape, pattern::any_input()}, concat_predicate);
    auto rhs_concat = std::make_shared<pattern::op::Or>(OutputVector{rhs_concat_0, rhs_concat_1});

    auto rhs_reshape_or_concat = std::make_shared<pattern::op::Or>(OutputVector{rhs_reshape, rhs_concat});

    auto rhs_bea_scalar = SCALAR_INPUT;
    auto rhs_bea = pattern::wrap_type<op::util::BinaryElementwiseArithmetic>({rhs_reshape_or_concat, rhs_bea_scalar},
                                                                             pattern::consumers_count(1));

    auto rhs_bea_or_concat = std::make_shared<pattern::op::Or>(OutputVector{rhs_reshape_or_concat, rhs_bea});
    // END: symmetrical patterns for MatMul inputs

    auto matmul =
        pattern::wrap_type<op::v0::MatMul>({lhs_bea_or_concat, rhs_bea_or_concat}, pattern::consumers_count(1));

    auto add = pattern::wrap_type<op::util::BinaryElementwiseArithmetic>(
        OutputVector{matmul, pattern::any_input()},
        [](ov::Output<Node> out) -> bool {
            // FIXME: two modes -- reducing and expanding from Reshape fusion
            // TODO: other input should be Reshape with Concat pattern -- then we can modify it to
            // FIXME: modify Reshapes pattern to reflect shrinking / expanding of dimensions !!!
            if (!pattern::consumers_count(1)(out))
                return false;
            auto input_0_pshape = out.get_node_shared_ptr()->get_input_partial_shape(0);
            auto input_1_pshape = out.get_node_shared_ptr()->get_input_partial_shape(1);
            auto output_pshape = out.get_partial_shape();
            ov::TensorLabel output_labels, input_0_labels, input_1_labels;
            if (get_labels(input_0_pshape, input_0_labels) && get_labels(input_1_pshape, input_1_labels) &&
                get_labels(output_pshape, output_labels))
                return are_unique_and_equal_labels(input_0_labels, output_labels) ||
                       are_unique_and_equal_labels(input_1_labels, output_labels);
            else
                return false;
        });

    auto matmul_or_add = std::make_shared<pattern::op::Or>(OutputVector{matmul, add});
    auto final_reshape =
        pattern::wrap_type<op::v1::Reshape>({matmul_or_add, pattern::any_input()},
                                            pattern::op::as_value_predicate([](std::shared_ptr<Node> n) -> bool {
                                                return reshape_keeps_last_two_dims(n);
                                            }));

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        std::cout << " MATCHED! " << std::endl;

        const auto& pm = m.get_pattern_map();
        const auto& vm = m.get_pattern_value_map();
        std::vector<Node*> nodes_for_revalidation{pm.at(matmul).get()};
        // reshapes check: BEGIN
        // reshape_keeps_last_two_dims checks were already applied for all Reshapes in the pattern predicates
        auto in_reshape_0 = pm.at(lhs_reshape);
        auto in_reshape_1 = pm.at(rhs_reshape);
        auto out_reshape = pm.at(final_reshape);
        if (!batches_are_equal(in_reshape_0, in_reshape_1) ||
            !batches_are_equal(in_reshape_0->get_output_partial_shape(0), out_reshape->get_input_partial_shape(0)) ||
            !batches_are_equal(in_reshape_0->get_input_partial_shape(0),
                               out_reshape->get_output_partial_shape(0),
                               true)) {
            return false;
        }
        // reshapes check: END

        // preventing wrong matches
        if (vm.count(lhs_concat) && !ov::as_type_ptr<ov::op::v0::Concat>(pm.at(lhs_concat)))
            return false;
        if (vm.count(rhs_concat) && !ov::as_type_ptr<ov::op::v0::Concat>(pm.at(rhs_concat)))
            return false;

        pull_reshape_through_optional_concat_and_bea(vm,
                                                     lhs_concat,
                                                     lhs_bea,
                                                     in_reshape_0,
                                                     pm.at(matmul)->input(0),
                                                     nodes_for_revalidation);
        pull_reshape_through_optional_concat_and_bea(vm,
                                                     rhs_concat,
                                                     rhs_bea,
                                                     in_reshape_1,
                                                     pm.at(matmul)->input(1),
                                                     nodes_for_revalidation);
        ov::replace_output_update_name(out_reshape->output(0), out_reshape->input_value(0));

        for (auto& node : nodes_for_revalidation)
            node->validate_and_infer_types();
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(final_reshape, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
