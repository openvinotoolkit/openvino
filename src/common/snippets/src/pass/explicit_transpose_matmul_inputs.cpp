// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "snippets/pass/explicit_transpose_matmul_inputs.hpp"
#include "snippets/pass/transpose_decomposition.hpp"
#include "snippets/op/subgraph.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::snippets::pass::ExplicitTransposeMatMulInputs::ExplicitTransposeMatMulInputs() {
    MATCHER_SCOPE(ExplicitTransposeMatMulInputs);

    auto m_matmul0 = std::make_shared<ov::opset1::MatMul>(
            ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape()),
            ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape()));

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_matmul0, matcher_name),
        [=](ov::pass::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::ExplicitTransposeMatMulInputs")
        auto root = m.get_match_root();
        bool rewritten = false;

        auto matmul0 = ov::as_type_ptr<ov::opset1::MatMul>(root);
        if (!matmul0)
            return false;

        for (size_t i = 0; i < matmul0->get_input_size(); i++) {
            if (i == 0 && !matmul0->get_transpose_a())
                continue;
            if (i == 1 && !matmul0->get_transpose_b())
                continue;

            auto parent1 = matmul0->get_input_node_shared_ptr(i);
            auto transpose1 = ov::as_type_ptr<ov::opset1::Transpose>(parent1);
            while (!transpose1 && !ov::is_type<ov::opset1::Parameter>(parent1)) {
                // We can set supported order and transposed_b(false) only if ops have scalar shapes to avoid shape mismatching
                const auto parent_count = parent1->inputs().size();
                bool are_weights_scalar = true;
                for (size_t j = 1; j < parent_count; ++j) {
                    are_weights_scalar = are_weights_scalar && ov::shape_size(parent1->get_input_shape(j)) == 1;
                }
                if (!are_weights_scalar)
                    break;

                parent1 = parent1->get_input_node_shared_ptr(0);
                transpose1 = ov::as_type_ptr<ov::opset1::Transpose>(parent1);
            }
            if (!transpose1)
                continue;

            const auto transpose_pattern = ov::as_type_ptr<ov::opset1::Constant>(transpose1->get_input_node_shared_ptr(1));
            if (!transpose_pattern)
                continue;

            auto transposed_order = transpose_pattern->cast_vector<int32_t>();
            std::swap(*transposed_order.rbegin(), *(transposed_order.rbegin() + 1));
            if (pass::TransposeDecomposition::supported_cases.count(transposed_order) == 0)
                continue;

            auto new_transpose_order = std::make_shared<ov::opset1::Constant>(transpose_pattern->get_element_type(),
                                                                                  ov::Shape{4},
                                                                                  transposed_order);
            new_transpose_order->set_friendly_name(transpose_pattern->get_friendly_name());
            ov::copy_runtime_info(transpose_pattern, new_transpose_order);
            transpose1->set_argument(1, new_transpose_order);
            if (i == 0) {
                matmul0->set_transpose_a(false);
            } else {
                matmul0->set_transpose_b(false);
            }
            rewritten |= true;
        }

        return rewritten;
    });
}
