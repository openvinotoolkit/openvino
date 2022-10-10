// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/variadic_split.hpp"
#include "ngraph/node.hpp"

#include <ngraph/pattern/op/wrap_type.hpp>

#include "low_precision/network_helper.hpp"
#include "itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

VariadicSplitTransformation::VariadicSplitTransformation(const Params& params) : SplitTransformation(params) {
    MATCHER_SCOPE(VariadicSplitTransformation);
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

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
