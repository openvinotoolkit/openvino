// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_subtract.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ngraph::pass::ConvertSubtract::ConvertSubtract() {
    auto sub = ngraph::pattern::wrap_type<ngraph::opset1::Subtract>();

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto sub = std::dynamic_pointer_cast<ngraph::opset1::Subtract> (m.get_match_root());
        if (!sub) {
            return false;
        }

        auto neg = std::make_shared<ngraph::opset1::Multiply>(sub->input(1).get_source_output(),
                                                              opset1::Constant::create(sub->get_input_element_type(1), Shape{1}, {-1}));

        auto add = std::make_shared<ngraph::opset1::Add>(sub->input(0).get_source_output(), neg);

        add->set_friendly_name(sub->get_friendly_name());
        ngraph::copy_runtime_info(sub, {neg, add});
        ngraph::replace_node(sub, add);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(sub, "ConvertSubtract");
    this->register_matcher(m, callback);
}