// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/preprocessing/mean_image_or_value.hpp"

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

using namespace ngraph;

ngraph::pass::AddMeanSubtract::AddMeanSubtract(const MeanMap& inputInfoMap) {
    // RUN_ON_FUNCTION_SCOPE(AddMeanSubtract);
    auto label = ngraph::pattern::wrap_type<ngraph::opset3::Parameter>();

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto param = std::dynamic_pointer_cast<ngraph::opset3::Parameter>(m.get_match_root());
        if (!param) {
            return false;
        }

        auto it = inputInfoMap.find(param->get_friendly_name());
        if (it == inputInfoMap.end()) {
            return false;
        }

        auto mean_const = it->second;
        NGRAPH_CHECK(mean_const->get_element_type() == ngraph::element::f32,
                     "Mean for ",
                     param->get_friendly_name(),
                     " must have f32 type");

        auto copy_param = param->clone_with_new_inputs({});
        auto sub = std::make_shared<ngraph::opset3::Subtract>(copy_param, mean_const);

        ngraph::replace_node(param, sub);
        sub->set_argument(0, param);

        // Return true as the root node was changed
        return true;
    };

    // Register pattern with Parameter operation as a pattern root node
    auto m = std::make_shared<ngraph::pattern::Matcher>(label, "AddMeanSubtract");
    // Register Matcher
    register_matcher(m, callback);
}
