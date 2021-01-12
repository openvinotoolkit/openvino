// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/mish_decomposition.hpp"

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include <memory>
#include <vector>

NGRAPH_RTTI_DEFINITION(vpu::MishDecomposition, "MishDecomposition", 0);

namespace vpu {

MishDecomposition::MishDecomposition() {
    const auto mishPattern = ngraph::pattern::wrap_type<ngraph::opset5::Mish>();

    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher &matcher) {
        const auto& mish = ngraph::as_type_ptr<ngraph::opset5::Mish>(matcher.get_match_root());

        if (!mish || transformation_callback(mish)) {
            return false;
        }

        const auto inputType = mish->input_value(0).get_element_type();
        const auto addConst = ngraph::opset5::Constant::create(inputType, ngraph::Shape{}, {1.0f});

        const auto exp = std::make_shared<ngraph::opset5::Exp>(mish->input_value(0));
        const auto add = std::make_shared<ngraph::opset5::Add>(exp, addConst);
        const auto log = std::make_shared<ngraph::opset5::Log>(add);
        const auto tanh = std::make_shared<ngraph::opset5::Tanh>(log);
        const auto mul = std::make_shared<ngraph::opset5::Multiply>(mish->input_value(0), tanh);

        mul->set_friendly_name(mish->get_friendly_name());
        ngraph::copy_runtime_info(mish, {addConst, exp, add, log, tanh, mul});
        ngraph::replace_node(mish, mul);

        return true;
    };

    const auto matcher = std::make_shared<ngraph::pattern::Matcher>(mishPattern, "MishDecomposition");
    register_matcher(matcher, callback);
}

}  // namespace vpu

