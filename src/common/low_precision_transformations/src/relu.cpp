// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/relu.hpp"

#include <algorithm>
#include <memory>
#include <string>

#include "itt.hpp"
#include "openvino/util/log.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"

namespace ov {
namespace pass {
namespace low_precision {

ReluTransformation::ReluTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(ReluTransformation);
    auto matcher = pattern::wrap_type<ov::opset1::Relu>({ pattern::wrap_type<ov::opset1::Multiply>() });

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

bool ReluTransformation::transform(ov::pass::pattern::Matcher &m) {
    std::shared_ptr<Node> relu = m.get_match_root();
    if (!canBeTransformed(relu)) {
        return false;
    }

    relu = NetworkHelper::separateInStandaloneBranch(relu, defaultPrecisions);
    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(relu, defaultPrecisions, 0);
    const auto newOperation = moveDequantizationAfter(relu, dequantization);

    OPENVINO_DEBUG("LPT: done: ", newOperation);
    return true;
}

bool ReluTransformation::isPrecisionPreserved(std::shared_ptr<Node> op) const noexcept {
    return true;
}

bool ReluTransformation::canBeTransformed(const std::shared_ptr<Node>& op) const {
    if (!LayerTransformation::canBeTransformed(op)) {
        return false;
    }

    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(op, defaultPrecisions, 0);
    if (dequantization.empty() || (dequantization.subtract != nullptr)) {
        return false;
    }

    const auto scales = dequantization.multiplyConstant->cast_vector<float>();
    if (std::any_of(scales.begin(), scales.end(), [](const float value) { return value < 0.f; })) {
        return false;
    }

    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ov
