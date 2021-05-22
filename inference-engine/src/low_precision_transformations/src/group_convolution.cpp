// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/group_convolution.hpp"

#include <memory>
#include <string>
#include <vector>

#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

GroupConvolutionTransformation::GroupConvolutionTransformation(const Params& params) : ConvolutionTransformation(params) {
}

void GroupConvolutionTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addSingleNodePattern<opset1::GroupConvolution>(pass, context);
}

bool GroupConvolutionTransformation::isQuantized(std::shared_ptr<Node> layer) const noexcept {
    return WeightableLayerTransformation::isQuantized(layer, true);
}

bool GroupConvolutionTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    auto convolution = m.get_match_root();

    if (!GroupConvolutionTransformation::canBeTransformed(context, convolution)) {
        return false;
    }

    ConvolutionTransformation::transform(context, m);
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
