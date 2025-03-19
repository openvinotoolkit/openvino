// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/space_to_batch.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/util/log.hpp"
#include "openvino/op/space_to_batch.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "low_precision/network_helper.hpp"

namespace ov {
namespace pass {
namespace low_precision {

SpaceToBatchTransformation::SpaceToBatchTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(SpaceToBatchTransformation);
    auto matcher = pattern::wrap_type<ov::op::v1::SpaceToBatch>();

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

bool SpaceToBatchTransformation::canBeTransformed(const std::shared_ptr<Node>& op) const {
    if (!LayerTransformation::canBeTransformed(op)) {
        return false;
    }

    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(op, defaultPrecisions);
    if (dequantization.empty()) {
        return false;
    }

    return dequantization.isPerTensor();
}

bool SpaceToBatchTransformation::transform(ov::pass::pattern::Matcher& m) {
    if (!canBeTransformed(m.get_match_root())) {
        return false;
    }

    const std::shared_ptr<Node> op = NetworkHelper::separateInStandaloneBranch(m.get_match_root(), defaultPrecisions);
    const auto newOperation = moveDequantizationAfter(op, NetworkHelper::getDequantization(op, defaultPrecisions));

    OPENVINO_DEBUG("LPT: done: ", newOperation);
    return true;
}

bool SpaceToBatchTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ov
