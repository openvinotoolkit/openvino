// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_negative.hpp"
#include "transformations/itt.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertNegative, "ConvertNegative", 0);

ngraph::pass::ConvertNegative::ConvertNegative() {
    auto neg = ngraph::pattern::wrap_type<ngraph::opset1::Negative>();
#if GraphGen(OV_GEN_NGRAPH_PASS(ConvertNegative, callback))
    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        OV_ITT_IE_TRANSFORM_CALLBACK(m, "callback")
        auto neg = std::dynamic_pointer_cast<ngraph::opset1::Negative> (m.get_match_root());
        if (!neg) {
            return false;
        }

        auto mul = std::make_shared<ngraph::opset1::Multiply>(neg->input(0).get_source_output(),
                                                              opset1::Constant::create(neg->get_element_type(), Shape{1}, {-1}));
        mul->set_friendly_name(neg->get_friendly_name());
        ngraph::copy_runtime_info(neg, mul);
        ngraph::replace_node(neg, mul);
        return true;
    };
#else
    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher & m) -> bool {
        return false;
    };
#endif
    auto m = std::make_shared<ngraph::pattern::Matcher>(neg, "ConvertNegative");
    this->register_matcher(m, callback);
}