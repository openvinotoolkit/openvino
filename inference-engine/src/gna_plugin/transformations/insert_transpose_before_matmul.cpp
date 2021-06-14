// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/insert_transpose_before_matmul.hpp"

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>

using namespace GNAPluginNS;

NGRAPH_RTTI_DEFINITION(InsertTransposeBeforeMatmul, "InsertTransposeBeforeMatmul", 0);

InsertTransposeBeforeMatmul::InsertTransposeBeforeMatmul() {
    auto reshape = ngraph::pattern::wrap_type<ngraph::opset7::Reshape>({ngraph::pattern::any_input(),
                                                                        ngraph::pattern::any_input()},
                                                                        ngraph::pattern::rank_equals(2));
    auto matmul1 = ngraph::pattern::wrap_type<ngraph::opset7::MatMul>({ngraph::pattern::any_input(), reshape});
    auto matmul2 = ngraph::pattern::wrap_type<ngraph::opset7::MatMul>({reshape, ngraph::pattern::any_input()});
    auto root = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{matmul1, matmul2});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        auto& pattern_map = m.get_pattern_value_map();
        auto reshape_node = pattern_map.at(reshape).get_node_shared_ptr();
        auto reshape_in_shape = reshape_node->get_input_shape(0);
        auto reshape_out_shape = reshape_node->get_output_shape(0);
        if (reshape_in_shape.front() == reshape_out_shape.front()) {
            return false;
        }

        if (reshape_out_shape[0] == 1 || reshape_out_shape[1] == 1) {
            return false;
        }

        size_t min, max;
        std::tie(min, max) = std::minmax(reshape_out_shape[0], reshape_out_shape[1]);
        if (min > 8 || max % 8 != 0) return false;

        auto consumers = reshape_node->output(0).get_target_inputs();
        auto matmul_node = consumers.begin()->get_node()->shared_from_this();

        auto transpose_order = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, std::vector<size_t>{1, 0});
        auto transpose = register_new_node<ngraph::opset7::Transpose>(reshape_node, transpose_order);
        transpose->set_friendly_name(matmul_node->get_friendly_name() + "/in_transpose");

        auto transpose_out_shape = transpose->output(0).get_shape();
        std::swap(transpose_out_shape[0], transpose_out_shape[1]);
        auto reshapeConstAfter = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
                                                                            ngraph::Shape{2},
                                                                            transpose_out_shape);
        auto reshapeAfter = std::make_shared<ngraph::opset7::Reshape>(transpose, reshapeConstAfter, false);
        reshapeAfter->set_friendly_name(matmul_node->get_friendly_name() + "/reshape_after_transpose");

        for (auto input : consumers) {
            input.replace_source_output(reshapeAfter);
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(root, "InsertTransposeBeforeMatmul");
    this->register_matcher(m, callback);
}
