// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/broadcast.hpp"

#include <memory>
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "low_precision/network_helper.hpp"
#include "itt.hpp"

using namespace ov::pass::low_precision;

BroadcastTransformation::BroadcastTransformation(const Params& params) : TransparentBaseTransformation(params) {
    MATCHER_SCOPE(BroadcastTransformation);
    auto matcher = pattern::wrap_type<ov::opset1::Broadcast>({
        pattern::wrap_type<ov::opset1::Multiply>(),
        ov::pass::pattern::any_input(),
        ov::pass::pattern::any_input() });

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
    if (dequantization.multiply != nullptr) {
        if (!dequantization.isPerTensor()) {
            return false;
        }
    }

    if (dequantization.subtract != nullptr) {
        if (!dequantization.isPerTensor()) {
            return false;
        }
    }

    return true;
}
