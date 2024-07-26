// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/broadcast.hpp"

#include <memory>

#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "low_precision/network_helper.hpp"

#include "itt.hpp"

using namespace ov::pass::low_precision;

BroadcastTransformation::BroadcastTransformation(const Params& params) : TransparentBaseTransformation(params) {
    MATCHER_SCOPE(BroadcastTransformation);
    auto broadcast1 = pattern::wrap_type<ov::opset1::Broadcast>({
        pattern::wrap_type<ov::opset1::Multiply>(),
        ov::pass::pattern::any_input(),
        ov::pass::pattern::any_input() });

    auto broadcast3 = pattern::wrap_type<ov::opset3::Broadcast>({
        pattern::wrap_type<ov::opset1::Multiply>(),
        ov::pass::pattern::any_input(),
        ov::pass::pattern::any_input() });

    const auto matcher = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{ broadcast1, broadcast3 });

    ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool BroadcastTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<ov::Node> layer) const {
    if (!LayerTransformation::canBeTransformed(context, layer)) {
        return false;
    }

    const auto& dequantization = NetworkHelper::getDequantization(layer, defaultPrecisions);
    if (dequantization.empty()) {
        return false;
    }

    if (dequantization.isPerTensor()) {
        return true;
    }

    const auto& inputShape = layer->get_input_partial_shape(0);
    if (inputShape.rank().is_dynamic() || inputShape[dequantization.channelDimIndex].is_dynamic()) {
        return false;
    }

    const auto targetShapeConstant = ov::as_type_ptr<ov::opset1::Constant>(layer->get_input_node_shared_ptr(1));
    const auto& targetShape = targetShapeConstant->cast_vector<int64_t>();
    if (targetShape[dequantization.channelDimIndex] != inputShape[dequantization.channelDimIndex].get_length()) {
        return false;
    }

    const auto axesMappingConstant = ov::as_type_ptr<ov::opset1::Constant>(layer->get_input_node_shared_ptr(2));
    const auto& axesMapping = axesMappingConstant->cast_vector<int64_t>();
    if (static_cast<size_t>(axesMapping[dequantization.channelDimIndex]) != dequantization.channelDimIndex) {
        return false;
    }

    return true;
}
