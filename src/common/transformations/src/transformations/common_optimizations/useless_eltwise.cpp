// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/useless_eltwise.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::EliminateUselessMul, "EliminateUselessMul", 0);

ngraph::pass::EliminateUselessMul::EliminateUselessMul() {
    MATCHER_SCOPE(EliminateUselessMul);
    // Remove Multiply if performs multiplication to `1`
    const auto input_1 = ngraph::pattern::any_input();
    const auto input_2 = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
    const auto mul = ngraph::pattern::wrap_type<ngraph::opset8::Multiply>({input_1, input_2},
                                                                                 pattern::consumers_count(1));

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& patternValue = m.get_pattern_value_map();

        const auto& m_const = patternValue.at(input_2);

        const auto& constNode =
                ngraph::as_type_ptr<ngraph::opset8::Constant>(m_const.get_node_shared_ptr());

        const auto& constVec = constNode->cast_vector<int64_t>();

        if (constVec.size() != 1) {
            return false;
        }

        if (constVec.front() != 1) {
            return false;
        }

        auto& m_input = patternValue.at(input_1);
        const auto& m_mul = patternValue.at(mul);

        ngraph::copy_runtime_info(m_mul.get_node_shared_ptr(), m_input.get_node_shared_ptr());
        m_input.replace(m_mul);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul, "EliminateUselessMul");
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::EliminateUselessDiv, "EliminateUselessDiv", 0);

ngraph::pass::EliminateUselessDiv::EliminateUselessDiv() {
    MATCHER_SCOPE(EliminateUselessDiv);
    // Remove Divide if it performs division to `1`
    const auto input_1 = ngraph::pattern::any_input();
    const auto input_2 = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
    const auto div = ngraph::pattern::wrap_type<ngraph::opset8::Divide>({input_1, input_2},
                                                                          pattern::consumers_count(1));

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& patternValue = m.get_pattern_value_map();

        const auto& m_const = patternValue.at(input_2);

        const auto& constNode =
                ngraph::as_type_ptr<ngraph::opset8::Constant>(m_const.get_node_shared_ptr());

        const auto& constVec = constNode->cast_vector<int64_t>();

        if (constVec.size() != 1) {
            return false;
        }

        if (constVec.front() != 1) {
            return false;
        }

        auto& m_input = patternValue.at(input_1);
        const auto& m_div = patternValue.at(div);

        ngraph::copy_runtime_info(m_div.get_node_shared_ptr(), m_input.get_node_shared_ptr());
        m_input.replace(m_div);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(div, "EliminateUselessDiv");
    register_matcher(m, callback);
}
