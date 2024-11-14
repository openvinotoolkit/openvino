// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/max_pool.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/util/log.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "low_precision/network_helper.hpp"

namespace ov {
namespace pass {
namespace low_precision {

MaxPoolTransformation::MaxPoolTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(MaxPoolTransformation);
    auto matcher = pattern::wrap_type<opset1::MaxPool>({ pattern::wrap_type<opset1::Multiply>() });

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

bool MaxPoolTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    if (!LayerTransformation::canBeTransformed(context, op)) {
        return false;
    }

    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(op, defaultPrecisions);
    if (dequantization.empty()) {
        return false;
    }

    const std::vector<float> scales = dequantization.multiplyConstant->cast_vector<float>();
    if (std::any_of(scales.begin(), scales.end(), [](const float value) { return value < 0.0; })) {
        return false;
    }

    return true;
}

bool MaxPoolTransformation::transform(TransformationContext& context, ov::pass::pattern::Matcher &m) {
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const std::shared_ptr<Node> pooling = NetworkHelper::separateInStandaloneBranch(m.get_match_root(), defaultPrecisions);
    const auto newOperation = moveDequantizationAfter(context, pooling, NetworkHelper::getDequantization(pooling, defaultPrecisions));

    OPENVINO_DEBUG("LPT: done: ", newOperation);
    return true;
}

bool MaxPoolTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ov
