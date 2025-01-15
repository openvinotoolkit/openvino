// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/pass/transform_if.hpp"

#include "default_opset.hpp"
#include "internal/op/conditional_block.hpp"
#include "internal/op/tensorarray_write.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/fold_subgraph_empty_inputs.hpp"

using namespace std;
using namespace ov;
using namespace ov::pass;
using namespace ov::frontend::paddle::op::default_opset;

// Transform Paddle "conditional_block" to OpenVINO If op.
// The conditional_block only has "then" branch, while If op requires both "then" and "else" branch the same time.
// Thus a "pass-through" model is built on purpose for "else" branch with the same outputs as "then" branch.
ov::frontend::paddle::pass::TransformIf::TransformIf(std::vector<std::shared_ptr<Model>> funcs) {
    const auto cond_label = pattern::wrap_type<ov::op::internal::ConditionalBlock>();

    matcher_pass_callback callback = [funcs](pattern::Matcher& m) -> bool {
        const auto conditional_block = ov::as_type_ptr<ov::op::internal::ConditionalBlock>(m.get_match_root());
        const auto mask_idx = conditional_block->get_input_size() - 1;
        const auto cond = conditional_block->get_input_node_shared_ptr(mask_idx);

        if (!conditional_block || !cond) {
            return false;
        }

        // build_if_node
        const auto then_idx = conditional_block->get_subblock_index();
        const auto& then_branch = funcs[then_idx];
        const auto& then_params = then_branch->get_parameters();

        // make a pass-through else branch, as
        // openvino If requires both then and else branch at the same time.
        ParameterVector params;
        ResultVector results;
        for (size_t i = 0; i < then_branch->get_output_size(); i++) {
            const auto param = std::make_shared<Parameter>(then_branch->get_output_element_type(i),
                                                           then_branch->get_output_partial_shape(i));
            param->set_friendly_name(then_branch->get_output_op(i)->get_output_tensor(0).get_any_name());
            params.push_back(param);
            const auto result = std::make_shared<Result>(param);
            results.push_back(result);
        }
        const auto else_branch = std::make_shared<Model>(results, params);
        const auto& else_params = else_branch->get_parameters();

        auto if_node = std::make_shared<If>(cond);
        ov::pass::disable_fold_subgraph_empty_inputs(if_node);
        if_node->set_then_body(then_branch);
        if_node->set_else_body(else_branch);

        const auto then_branch_inputs_from_parent = conditional_block->get_inputs_from_parent();
        OPENVINO_ASSERT(then_branch_inputs_from_parent.size() == then_params.size(),
                        "Number of inputs to 'then_branch' is invalid. Expected " +
                            std::to_string(then_branch_inputs_from_parent.size()) + ", actual " +
                            std::to_string(then_params.size()));
        auto then_param = then_params.cbegin();
        for (const auto& from_parent : then_branch_inputs_from_parent) {
            if_node->set_input(from_parent, *then_param, nullptr);
            then_param++;
        }

        for (const auto& else_param : else_params) {
            bool found = false;
            for (const auto& from_parent : then_branch_inputs_from_parent) {
                if (from_parent.get_any_name() == else_param->get_friendly_name()) {
                    if_node->set_input(from_parent, nullptr, else_param);
                    found = true;
                    break;
                }
            }
            // the output generate from the body, make a default value
            if (!found) {
                auto ps = else_param->get_partial_shape();
                const auto placeholder = Constant::create(else_param->get_element_type(), ps.get_min_shape(), {0});
                if_node->set_input(placeholder, nullptr, else_param);
            }
        }

        auto else_results = else_branch->get_results();
        auto then_results = then_branch->get_results();
        for (size_t i = 0; i < else_results.size(); i++) {
            if_node->set_output(then_results[i], else_results[i]);
        }
        replace_node(conditional_block, if_node);
        if_node->set_friendly_name(conditional_block->get_friendly_name());

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(cond_label, "condtionalblock_if");
    this->register_matcher(m, callback);
}
