// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/batch_to_space.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/util/log.hpp"

#include "openvino/op/batch_to_space.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "low_precision/network_helper.hpp"

namespace ov {
namespace pass {
namespace low_precision {

BatchToSpaceTransformation::BatchToSpaceTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(BatchToSpaceTransformation);
    auto matcher = pattern::wrap_type<ov::op::v1::BatchToSpace>();

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

bool BatchToSpaceTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    if (!LayerTransformation::canBeTransformed(context, op)) {
        return false;
    }

    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(op, defaultPrecisions);
    if (dequantization.empty()) {
        return false;
    }

    return dequantization.isPerTensor();
}

bool BatchToSpaceTransformation::transform(TransformationContext& context, ov::pass::pattern::Matcher& m) {
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const std::shared_ptr<Node> op = NetworkHelper::separateInStandaloneBranch(m.get_match_root(), defaultPrecisions);
    const auto newOperation = moveDequantizationAfter(context, op, NetworkHelper::getDequantization(op, defaultPrecisions));

    OPENVINO_DEBUG("LPT: done: ", newOperation);
    return true;
}

bool BatchToSpaceTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ov
