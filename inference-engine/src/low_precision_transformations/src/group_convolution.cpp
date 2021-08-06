// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/group_convolution.hpp"

#include <memory>
#include <string>
#include <vector>

#include <ngraph/pattern/op/wrap_type.hpp>
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::GroupConvolutionTransformation, "GroupConvolutionTransformation", 0);

GroupConvolutionTransformation::GroupConvolutionTransformation(const Params& params) : ConvolutionTransformation(params) {
    auto matcher = pattern::wrap_type<opset1::GroupConvolution>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "GroupConvolutionTransformation");
    this->register_matcher(m, callback);
}

bool GroupConvolutionTransformation::isQuantized(const std::shared_ptr<const Node>& layer) const noexcept {
    return GroupConvolutionTransformation::isQuantizedStatic(layer);
}

bool GroupConvolutionTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) {
    auto convolution = m.get_match_root();

    if (!GroupConvolutionTransformation::canBeTransformed(context, convolution)) {
        return false;
    }

    ConvolutionTransformation::transform(context, m);
    return true;
}

bool GroupConvolutionTransformation::isQuantizedStatic(const std::shared_ptr<const Node>& layer) noexcept {
    return WeightableLayerTransformation::isQuantizedStatic(layer, true);
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
