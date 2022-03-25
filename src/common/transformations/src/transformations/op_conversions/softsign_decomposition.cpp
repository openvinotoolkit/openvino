// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/op_conversions/softsign_decomposition.hpp>

#include "itt.hpp"

ngraph::pass::SoftSignDecomposition::SoftSignDecomposition() {
    MATCHER_SCOPE(SoftSignDecomposition);
    auto softsign = pattern::wrap_type<ngraph::opset9::SoftSign>();
    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto m_softsign = m.get_match_root();

        if (transformation_callback(m_softsign)) {
            return false;
        }

        Output<Node> input = m_softsign->input_value(0);
        auto data_type = m_softsign->get_input_element_type(0);
        auto abs = std::make_shared<ngraph::opset9::Abs>(input);
        auto constant = ngraph::opset9::Constant::create(data_type, ngraph::Shape{1}, {1});
        auto add = std::make_shared<ngraph::opset9::Add>(abs, constant);
        auto div = std::make_shared<ngraph::opset9::Divide>(input, add);

        replace_node(m_softsign, div);
        copy_runtime_info(m_softsign, {abs, add, div});
        div->set_friendly_name(m_softsign->get_friendly_name());

        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(softsign, matcher_name);
    register_matcher(m, callback);
}
