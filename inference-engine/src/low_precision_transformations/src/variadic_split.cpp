// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/variadic_split.hpp"
#include "ngraph/node.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

VariadicSplitTransformation::VariadicSplitTransformation(const Params& params) : SplitTransformation(params) {
    auto matcher = make_op_pattern<opset1::VariadicSplit>({
        make_op_label<opset1::Multiply>(),
        make_op_label<opset1::Constant>(),
        make_op_label<opset1::Constant>() });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (!op || m_transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "VariadicSplitTransformation");
    this->register_matcher(m, callback);
}

void VariadicSplitTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(pass,
               context,
               make_op_pattern<opset1::VariadicSplit>({
                    make_op_label<opset1::Multiply>(),
                    make_op_label<opset1::Constant>(),
                    make_op_label<opset1::Constant>() }));
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
