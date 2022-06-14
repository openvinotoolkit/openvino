// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/group_convolution.hpp"

#include <memory>
#include <string>
#include <vector>

#include <ngraph/pattern/op/wrap_type.hpp>
#include "low_precision/network_helper.hpp"
#include "itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

GroupConvolutionTransformation::GroupConvolutionTransformation(const Params& params) : ConvolutionTransformation(params) {
    MATCHER_SCOPE(GroupConvolutionTransformation);
    auto matcher = pattern::wrap_type<opset1::GroupConvolution>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        MATCHER_SCOPE_ENABLE(GroupConvolutionTransformation);
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool GroupConvolutionTransformation::isQuantized(const std::shared_ptr<const Node>& layer,
    const std::vector<ngraph::element::Type>& defaultPrecisions) const {
    return GroupConvolutionTransformation::isQuantizedStatic(layer, defaultPrecisions);
}

bool GroupConvolutionTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) {
    auto convolution = m.get_match_root();

    if (!GroupConvolutionTransformation::canBeTransformed(context, convolution)) {
        return false;
    }

    ConvolutionTransformation::transform(context, m);
    return true;
}

bool GroupConvolutionTransformation::isQuantizedStatic(const std::shared_ptr<const Node>& layer,
    const std::vector<ngraph::element::Type>& defaultPrecisions) {
    return WeightableLayerTransformation::isQuantizedStatic(layer, true, defaultPrecisions);
}

size_t GroupConvolutionTransformation::getInputChannels(const std::shared_ptr<ngraph::Node> conv) const {
    const auto groups = conv->get_input_partial_shape(1)[0];
    const auto channels = conv->get_input_partial_shape(1)[2];
    assert(channels.is_static() && groups.is_static());
    return channels.get_length() * groups.get_length();
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
