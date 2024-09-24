// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "low_precision/slice.hpp"

#include "itt.hpp"
#include "openvino/util/log.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/opsets/opset8.hpp"

#include "low_precision/network_helper.hpp"

namespace ov {
namespace pass {
namespace low_precision {

SliceTransformation::SliceTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(SliceTransformation);
    auto matcher = ov::pass::pattern::wrap_type<ov::opset8::Slice>();

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

bool SliceTransformation::transform(TransformationContext& context, ov::pass::pattern::Matcher& m) {
    if (!SliceTransformation::canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const auto strided_slice = NetworkHelper::separateInStandaloneBranch(m.get_match_root(), defaultPrecisions);
    const auto newOperation = moveDequantizationAfter(context, strided_slice, NetworkHelper::getDequantization(strided_slice, defaultPrecisions));

    OPENVINO_DEBUG("LPT: done: ", newOperation);
    return true;
}

bool SliceTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!LayerTransformation::canBeTransformed(context, operation)) {
        return false;
    }

    if (!ov::is_type<ov::opset8::Slice>(operation)) {
        return false;
    }

    const auto dequantization = NetworkHelper::getDequantization(operation);
    if (dequantization.empty()) {
        return false;
    }

    return dequantization.isPerTensor();
}

bool SliceTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}
} // namespace low_precision
} // namespace pass
} // namespace ov
