// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/preprocessing/std_scale.hpp"

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::AddStdScale, "AddStdScale", 0);

ngraph::pass::AddStdScale::AddStdScale(const ScaleMap& inputInfoMap) {
    // RUN_ON_FUNCTION_SCOPE(AddStdScale);
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

        auto scale_const = it->second;
        NGRAPH_CHECK(scale_const->get_element_type() == ngraph::element::f32, "Scale for ", param->get_friendly_name(), " must have f32 type");

        auto copy_param = param->clone_with_new_inputs({});
        auto div = std::make_shared<ngraph::opset3::Divide>(copy_param, it->second);

        ngraph::replace_node(param, div);
        div->set_argument(0, param);

        // Return true as the root node was changed
        return true;
    };

    // Register pattern with Parameter operation as a pattern root node
    auto m = std::make_shared<ngraph::pattern::Matcher>(label, "AddStdScale");
    // Register Matcher
    register_matcher(m, callback);
}
