// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/variadic_split.hpp"
#include "ngraph/node.hpp"

#include <ngraph/pattern/op/wrap_type.hpp>

#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::VariadicSplitTransformation, "VariadicSplitTransformation", 0);

VariadicSplitTransformation::VariadicSplitTransformation(const Params& params) : SplitTransformation(params) {
    auto matcher = pattern::wrap_type<opset1::VariadicSplit>({
        pattern::wrap_type<opset1::Multiply>(),
        pattern::wrap_type<opset1::Constant>(),
        pattern::wrap_type<opset1::Constant>() });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "VariadicSplitTransformation");
    this->register_matcher(m, callback);
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
