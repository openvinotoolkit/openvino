// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/group_convolution.hpp"

#include <memory>
#include <string>
#include <vector>

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "low_precision/network_helper.hpp"
#include "itt.hpp"

namespace ov {
namespace pass {
namespace low_precision {

GroupConvolutionTransformation::GroupConvolutionTransformation(const Params& params) : ConvolutionTransformation(params) {
    MATCHER_SCOPE(GroupConvolutionTransformation);
    auto matcher = pattern::wrap_type<ov::opset1::GroupConvolution>();

    ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(m);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool GroupConvolutionTransformation::isQuantized(const std::shared_ptr<const Node>& layer,
    const std::vector<ov::element::Type>& defaultPrecisions) const {
    return GroupConvolutionTransformation::isQuantizedStatic(layer, defaultPrecisions);
}

bool GroupConvolutionTransformation::transform(ov::pass::pattern::Matcher &m) {
    auto convolution = m.get_match_root();
    if (!WeightableLayerTransformation::canBeTransformed(convolution)) {
        return false;
    }
    return ConvolutionTransformation::transform(m);
}

bool GroupConvolutionTransformation::isQuantizedStatic(const std::shared_ptr<const Node>& layer,
    const std::vector<ov::element::Type>& defaultPrecisions) {
    return WeightableLayerTransformation::isQuantizedStatic(layer, true, defaultPrecisions);
}

size_t GroupConvolutionTransformation::getInputChannels(const std::shared_ptr<ov::Node> conv) const {
    const auto groups = conv->get_input_partial_shape(1)[0];
    const auto channels = conv->get_input_partial_shape(1)[2];
    assert(channels.is_static() && groups.is_static());
    return channels.get_length() * groups.get_length();
}

} // namespace low_precision
} // namespace pass
} // namespace ov
