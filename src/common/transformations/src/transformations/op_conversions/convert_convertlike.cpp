// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_convertlike.hpp"

#include <memory>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <vector>

#include "itt.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertConvertLike, "ConvertConvertLike", 0);

using namespace ngraph;

ngraph::pass::ConvertConvertLike::ConvertConvertLike() {
    MATCHER_SCOPE(ConvertConvertLike);

    auto convertlike = pattern::wrap_type<opset8::ConvertLike>();

    matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto cvtlike = std::dynamic_pointer_cast<opset8::ConvertLike>(m.get_match_root());
        if (!cvtlike) {
            return false;
        }

        auto like = cvtlike->input_value(1);
        const element::Type& dest_type = like.get_element_type();
        if (dest_type == element::dynamic || dest_type == element::undefined)
            return false;

        auto cvt = std::make_shared<opset8::Convert>(cvtlike->input_value(0), dest_type);

        cvt->set_friendly_name(cvtlike->get_friendly_name());
        copy_runtime_info(cvtlike, cvt);
        replace_node(cvtlike, cvt);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(convertlike, matcher_name);
    this->register_matcher(m, callback);
}
