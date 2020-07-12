// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/group_convolution.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <cassert>

#include "transformations/low_precision/network_helper.hpp"

// TODO: remove after debugging
#include <ngraph/pass/visualize_tree.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

GroupConvolutionTransformation::GroupConvolutionTransformation(const Params& params) : ConvolutionTransformation(params) {
}

void GroupConvolutionTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    // addPattern(
    //    pass,
    //    context,
    //    make_op_pattern<opset1::GroupConvolution>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::FakeQuantize>()}));

    // addPattern(
    //    pass,
    //    context,
    //    make_op_pattern<opset1::GroupConvolution>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::Multiply>() }));

    addSingleNodePattern<opset1::GroupConvolution>(pass, context);
}

// TODO: not completed
void GroupConvolutionTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    // TODO: per channel queantization is supported - fix later

    auto convolution = m.get_match_root();
    if (!GroupConvolutionTransformation::canBeTransformed(context, convolution)) {
        return;
    }

    // TODO: use new method without canBeTransformed
    ConvolutionTransformation::transform(context, m);
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
