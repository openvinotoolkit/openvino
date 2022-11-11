// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include "snippets/remarks.hpp"

#include "snippets/pass/matmul_transpose.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/validation_util.hpp>



ngraph::snippets::pass::MatMulTranspose::MatMulTranspose() {
    MATCHER_SCOPE(MatMulTranspose);

    auto m_matmul0 = std::make_shared<ngraph::opset1::MatMul>(
            ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
            ngraph::pattern::any_input(ngraph::pattern::has_static_shape()));

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(m_matmul0, matcher_name),
        [=](ngraph::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::MatMulTranspose")
        auto root = m.get_match_root();

        auto matmul0 = ngraph::as_type_ptr<ngraph::opset1::MatMul>(root);
        if (!matmul0 || !matmul0->get_transpose_b())
            return false;
        auto parent1 = matmul0->get_input_node_shared_ptr(1);
        auto transpose1 = ngraph::as_type_ptr<ngraph::opset1::Transpose>(parent1);
        while (!transpose1 && !ov::is_type<ngraph::opset1::Parameter>(parent1)) {
            parent1 = parent1->get_input_node_shared_ptr(0);
            transpose1 = ngraph::as_type_ptr<ngraph::opset1::Transpose>(parent1);
        }
        if (!transpose1)
            return false;

        const auto transpose_pattern = ngraph::as_type_ptr<ngraph::opset1::Constant>(transpose1->get_input_node_shared_ptr(1));
        // TODO: Reuse supported order from Transpose decomposition
        if (!transpose_pattern || transpose_pattern->cast_vector<int64_t>() != std::vector<int64_t>{0, 2, 1, 3})
            return false;

        auto new_transpose_order = std::make_shared<ngraph::opset1::Constant>(transpose_pattern->get_element_type(),
                                                                              ngraph::Shape{4},
                                                                              std::vector<int64_t>{0, 2, 3, 1});
        new_transpose_order->set_friendly_name(transpose_pattern->get_friendly_name());
        ngraph::copy_runtime_info(transpose_pattern, new_transpose_order);
        transpose1->set_argument(1, new_transpose_order);
        matmul0->set_transpose_b(false);

        return true;
    });
}
