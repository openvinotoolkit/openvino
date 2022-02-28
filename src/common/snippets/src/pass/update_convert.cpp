// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/update_convert.hpp"

#include <snippets/itt.hpp>
#include "remarks.hpp"

#include "snippets/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ngraph::snippets::pass::UpdateConvert::UpdateConvert() {
    MATCHER_SCOPE(UpdateConvert);

    const auto matcher = std::make_shared<ngraph::pattern::Matcher>(ngraph::pattern::wrap_type<ngraph::opset1::Convert>());
    const auto callback = [this](ngraph::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::FuseLoadAndConvert")
        auto root = m.get_match_root();
        if (transformation_callback(root)) {
            return false;
        }

        return true;
    };

    register_matcher(matcher, callback);
}

