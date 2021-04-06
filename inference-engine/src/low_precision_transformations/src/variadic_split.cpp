// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/variadic_split.hpp"
#include "ngraph/node.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {
VariadicSplitTransformation::VariadicSplitTransformation(const Params& params) : SplitTransformation(params) {}

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
