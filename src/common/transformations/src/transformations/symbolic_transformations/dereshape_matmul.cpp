// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/dereshape_matmul.hpp"

#include "itt.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/symbolic_transformations/utils.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::symbol::util;

namespace {
bool concat_predicate(ov::Output<ov::Node> output) {
    auto output_pshape = output.get_partial_shape();
    if (output_pshape.rank().is_dynamic() || output_pshape.size() <= 2)
        return false;
    const auto& concat = ov::as_type_ptr<ov::op::v0::Concat>(output.get_node_shared_ptr());
    if (!concat)
        return false;

    const auto norm_axis = ov::util::normalize(concat->get_axis(), output_pshape.rank().get_length());
    return norm_axis >= output_pshape.rank().get_length() - 2;
}

bool last_two_dims_are_equal(const ov::PartialShape& lhs, const ov::PartialShape& rhs) {
    if (lhs.rank().is_dynamic() || lhs.size() < 2)
        return false;
    if (rhs.rank().is_dynamic() || rhs.size() < 2)
        return false;
    auto lhs_dim = lhs.rbegin(), rhs_dim = rhs.rbegin();
    for (int i = 0; i < 2; ++i, lhs_dim++, rhs_dim++)
        if (!dims_are_equal(*lhs_dim, *rhs_dim))
            return false;
    return true;
}

bool reshape_keeps_last_two_dims(const std::shared_ptr<ov::Node>& op) {
    return last_two_dims_are_equal(op->get_input_partial_shape(0), op->get_output_partial_shape(0));
}

bool batches_are_equal(const ov::PartialShape& lhs, const ov::PartialShape& rhs, bool one_dim_can_differ = false) {
    if (lhs.rank().is_dynamic() || rhs.rank().is_dynamic() || lhs.size() != rhs.size())
        return false;
    size_t num_dims_differ = 0;
    for (size_t i = 0; i < lhs.size() - 2; ++i)
        num_dims_differ += !dims_are_equal(lhs[i], rhs[i]);
    return num_dims_differ <= (one_dim_can_differ ? 1 : 0);
}

bool batches_are_equal(const std::shared_ptr<ov::Node>& op_0, const std::shared_ptr<ov::Node>& op_1) {
    auto input_0 = op_0->get_input_partial_shape(0);
    auto input_1 = op_1->get_input_partial_shape(0);
    auto output_0 = op_0->get_output_partial_shape(0);
    auto output_1 = op_1->get_output_partial_shape(0);
    return batches_are_equal(input_0, input_1, true) && batches_are_equal(output_0, output_1);
}

void get_dims(const ov::Output<ov::Node>& source,
              const size_t& from,
              const size_t& to,
              const std::vector<std::shared_ptr<ov::Node>>& copy_rt_info_from,
              ov::NodeVector& dims) {
    std::vector<size_t> non_constant_ids;
    for (size_t i = from; i < to; ++i) {
        auto node = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(source, {i}, copy_rt_info_from);
        if (auto constant = ov::util::get_constant_from_source(node)) {
            node = constant;
        } else {
            non_constant_ids.push_back(i);
        }
        dims.push_back(node);
    }
}

ov::Output<ov::Node> get_target_shape_from_sources(const ov::Output<ov::Node>& batch_dims_source,
                                                   const ov::Output<ov::Node>& non_batch_dims_source,
                                                   const std::vector<std::shared_ptr<ov::Node>>& copy_rt_info_from) {
    ov::NodeVector dims;
    // batch dims here stand for MatMul batch dims -- leaving two last dims for Matrix Multiplication
    size_t num_batch_dims = batch_dims_source.get_partial_shape().size() - 2;
    get_dims(batch_dims_source, 0, num_batch_dims, copy_rt_info_from, dims);

    size_t non_batch_dims_start = non_batch_dims_source.get_partial_shape().size() - 2;
    get_dims(non_batch_dims_source, non_batch_dims_start, non_batch_dims_start + 2, copy_rt_info_from, dims);

    size_t num_non_const_nodes = 0;  // candidates for becoming a Constant -1 -- special value for Reshape pattern
    for (size_t curr_i = 0; curr_i + 1 < dims.size(); ++curr_i) {
        auto curr_node = dims[curr_i], next_node = dims[curr_i + 1];
        bool curr_is_const = ov::op::util::is_constant(curr_node), next_is_const = ov::op::util::is_constant(next_node);
        if (num_non_const_nodes == 0 && !curr_is_const && next_is_const) {
            curr_node = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            curr_is_const = true;
            num_non_const_nodes += 1;
        }
        if (num_non_const_nodes == 0 && !next_is_const && curr_is_const) {
            next_node = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            next_is_const = true;
            num_non_const_nodes += 1;
        }
        if (curr_is_const && next_is_const) {
            dims[curr_i] = nullptr;
            dims[curr_i + 1] = ov::op::util::make_try_fold<ov::op::v0::Concat>(ov::NodeVector{curr_node, next_node}, 0);
            ov::copy_runtime_info(copy_rt_info_from, dims[curr_i + 1]);
        }
    }
    dims.erase(std::remove_if(dims.begin(),
                              dims.end(),
                              [](const std::shared_ptr<ov::Node>& node) {
                                  return node == nullptr;
                              }),
               dims.end());
    auto target_shape = ov::op::util::make_try_fold<ov::op::v0::Concat>(dims, 0);
    ov::copy_runtime_info(copy_rt_info_from, target_shape);
    return target_shape->output(0);
}

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
        auto axis = (ov::util::normalize(concat_node->get_axis(), rank) == (rank - 1)) ? -1 : -2;

        auto idx_of_reshape_input = reshape_output == concat_node->input_value(0) ? 0 : 1;
        auto idx_of_non_reshape_input = static_cast<size_t>(!idx_of_reshape_input);

        auto target_shape_of_input = get_target_shape_from_sources(original_reshape->input_value(0),
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
            get_target_shape_from_sources(input_reshape->input_value(0), new_concat->output(0), {original_reshape});
        auto output_reshape = original_reshape->clone_with_new_inputs({new_concat->output(0), target_shape_of_output});
        ov::copy_runtime_info(original_reshape, output_reshape);

        if (vm.count(bea_label)) {
            auto bea_node = vm.at(bea_label).get_node_shared_ptr();
            auto idx_of_non_scalar_data = bea_node->input_value(0) == vm.at(concat_label) ? 0 : 1;
            bea_node->input(idx_of_non_scalar_data).replace_source_output(new_concat);
            nodes_for_revalidation.insert(nodes_for_revalidation.begin(), bea_node.get());
        } else {
            matmul_input.replace_source_output(new_concat);
        }
        ov::replace_output_update_name(concat_node->output(0), output_reshape->output(0));
    } else {
        // no Concat and it doesn't matter if BEA is present -- just delete reshape
        ov::replace_output_update_name(reshape_output, original_reshape->input_value(0));
    }
}
}  // namespace

#define IN_RESHAPE                                                                          \
    pattern::wrap_type<op::v1::Reshape>([](std::shared_ptr<Node> n) -> bool {               \
        return pattern::consumers_count(1)(n->output(0)) && reshape_keeps_last_two_dims(n); \
    });

#define SCALAR_INPUT                                                                        \
    pattern::any_input([](ov::Output<Node> out) {                                           \
        return out.get_partial_shape().is_static() && ov::shape_size(out.get_shape()) == 1; \
    });

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
            if (!pattern::consumers_count(1)(out))
                return false;
            auto input_0_pshape = out.get_node_shared_ptr()->get_input_partial_shape(0);
            auto input_1_pshape = out.get_node_shared_ptr()->get_input_partial_shape(1);
            auto output_pshape = out.get_partial_shape();
            ov::TensorSymbol output_symbols, input_0_symbols, input_1_symbols;
            if (get_symbols(input_0_pshape, input_0_symbols) && get_symbols(input_1_pshape, input_1_symbols) &&
                get_symbols(output_pshape, output_symbols)) {
                if (input_0_pshape.size() != 3 || input_1_pshape.size() != 3 || output_pshape.size() != 3)
                    return false;
                return are_unique_and_equal_symbols(input_0_symbols, output_symbols) ||
                       are_unique_and_equal_symbols(input_1_symbols, output_symbols);
            } else {
                return false;
            }
        });

    auto matmul_or_add = std::make_shared<pattern::op::Or>(OutputVector{matmul, add});
    auto final_reshape =
        pattern::wrap_type<op::v1::Reshape>({matmul_or_add, pattern::any_input()}, [](std::shared_ptr<Node> n) -> bool {
            return reshape_keeps_last_two_dims(n);
        });

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
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

        if (vm.count(add)) {
            const auto& in_reshape_0_in_pshape = in_reshape_0->get_input_partial_shape(0);
            if (in_reshape_0_in_pshape.size() != 4)
                return false;
            // we only allow MatMul -> Add pattern to be optimized in case of 4d -> 3d -> 4d DeReshaping
        }

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

        for (auto& node : nodes_for_revalidation)
            node->validate_and_infer_types();

        if (vm.count(add)) {
            auto add_node = pm.at(add);
            size_t matmul_port = (add_node->input_value(0) == vm.at(matmul) ? 0 : 1);
            size_t non_matmul_port = static_cast<size_t>(!matmul_port);

            auto first_batch_dim = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(
                add_node->input_value(non_matmul_port),
                {0},
                {in_reshape_0, in_reshape_1});
            auto divisor =
                ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(in_reshape_0->input_value(0),
                                                                                   {1},
                                                                                   {in_reshape_0, in_reshape_1});
            first_batch_dim = std::make_shared<ov::op::v1::Divide>(first_batch_dim, divisor, true);
            auto minus_one = ov::op::v0::Constant::create(element::i64, {1}, {-1});
            auto non_batch_dims = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(
                add_node->input_value(non_matmul_port),
                {1, 2},
                {in_reshape_0, in_reshape_1});
            auto pattern =
                std::make_shared<ov::op::v0::Concat>(OutputVector{first_batch_dim, minus_one, non_batch_dims}, 0);
            auto other_input_reshape =
                op::util::make_try_fold<ov::op::v1::Reshape>(add_node->input_value(non_matmul_port), pattern, true);
            add_node->input(non_matmul_port).replace_source_output(other_input_reshape->output(0));
            ov::copy_runtime_info({in_reshape_0, in_reshape_1},
                                  {first_batch_dim, minus_one, other_input_reshape, pattern});
            add_node->validate_and_infer_types();
        }
        ov::replace_output_update_name(out_reshape->output(0), out_reshape->input_value(0));
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(final_reshape, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::DeReshapeFullyConnected::DeReshapeFullyConnected() {
    MATCHER_SCOPE(DeReshapeFullyConnected);
    using namespace ov::op;
    using namespace ov::pass::pattern;

    auto transpose_a_false = [](const std::shared_ptr<Node>& node) -> bool {
        auto mm = as_type_ptr<v0::MatMul>(node);
        return mm && !mm->get_transpose_a();
    };

    auto input = wrap_type<v1::Reshape>({any_input(shape_matches("BATCHES_1...,Y")), any_input()},
                                        shape_matches("BATCHES_2...,Y"));
    auto converted = pattern::optional<v0::Convert>(input, consumers_count(1));
    auto mm_label = wrap_type<v0::MatMul>({converted, any_input(rank_equals(2))},
                                          consumers_count(1) && transpose_a_false && shape_matches("BATCHES_2...,Z"));
    auto output = wrap_type<v1::Reshape>({mm_label, any_input()}, shape_matches("BATCHES_1...,Z"));

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pm = m.get_pattern_map();
        const auto& in_reshape = pm.at(input);
        const auto& matmul = pm.at(mm_label);
        if (pm.count(converted)) {
            const auto& convert = pm.at(converted);
            convert->input(0).replace_source_output(in_reshape->input_value(0));
            convert->validate_and_infer_types();
        } else {
            matmul->input(0).replace_source_output(in_reshape->input_value(0));
        }

        ov::replace_output_update_name(m.get_match_root()->output(0), matmul->output(0));
        matmul->validate_and_infer_types();
        return true;
    };

    auto m = std::make_shared<Matcher>(output, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
