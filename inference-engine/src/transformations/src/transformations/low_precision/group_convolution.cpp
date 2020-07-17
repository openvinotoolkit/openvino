// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/group_convolution.hpp"

#include <memory>
#include <string>
#include <vector>

#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

GroupConvolutionTransformation::GroupConvolutionTransformation(const Params& params) : ConvolutionTransformation(params) {
}

void GroupConvolutionTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    // question to nGraph: why it doesn't work
    // addPattern(
    //    pass,
    //    context,
    //    make_op_pattern<opset1::GroupConvolution>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::FakeQuantize>()}));

    addSingleNodePattern<opset1::GroupConvolution>(pass, context);
}

void GroupConvolutionTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    auto convolution = m.get_match_root();
    if (!GroupConvolutionTransformation::canBeTransformed(context, convolution)) {
        return;
    }

    ConvolutionTransformation::transform(context, m);
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
