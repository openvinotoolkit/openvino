// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/broadcast.hpp"

#include <memory>

#include "itt.hpp"
#include "low_precision/network_helper.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov::pass::low_precision;

BroadcastTransformation::BroadcastTransformation(const Params& params) : TransparentBaseTransformation(params) {
    MATCHER_SCOPE(BroadcastTransformation);
    using namespace ov::pass::pattern;
    auto mul = wrap_type<ov::op::v1::Multiply>();
    auto matcher = wrap_type<ov::op::v3::Broadcast, ov::op::v1::Broadcast>({mul, any_input(), any_input()}) |
                   wrap_type<ov::op::v3::Broadcast>({mul, any_input()});

    ov::graph_rewrite_callback callback = [this](Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(m);
    };

    auto m = std::make_shared<Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool BroadcastTransformation::canBeTransformed(const std::shared_ptr<ov::Node>& layer) const {
    if (!TransparentBaseTransformation::canBeTransformed(layer)) {
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

    const auto& outputShape = layer->get_output_partial_shape(0);
    if (outputShape[dequantization.channelDimIndex] != inputShape[dequantization.channelDimIndex]) {
        return false;
    }

    const auto bcast = ov::as_type_ptr<ov::op::util::BroadcastBase>(layer);
    if (bcast == nullptr) {
        return false;
    }
    // axisMapping input affects the result only in case of explicit broadcast.
    if (bcast->get_broadcast_spec().m_type == ov::op::BroadcastType::EXPLICIT && bcast->get_input_size() == 3) {
        const auto axesMappingConstant = ov::as_type_ptr<ov::op::v0::Constant>(bcast->get_input_node_shared_ptr(2));
        if (!axesMappingConstant) {
            return false;
        }
        const auto& axesMapping = axesMappingConstant->cast_vector<size_t>();
        if (axesMapping[dequantization.channelDimIndex] != dequantization.channelDimIndex) {
            return false;
        }
    }

    return true;
}
