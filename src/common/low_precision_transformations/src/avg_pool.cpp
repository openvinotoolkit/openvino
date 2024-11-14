// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/avg_pool.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/util/log.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"

namespace ov {
namespace pass {
namespace low_precision {

AvgPoolTransformation::AvgPoolTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(AvgPoolTransformation);
    auto matcher = pattern::wrap_type<opset1::AvgPool>({ pattern::wrap_type<opset1::Multiply>() });

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

bool AvgPoolTransformation::transform(TransformationContext& context, ov::pass::pattern::Matcher &m) {
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const std::shared_ptr<Node> pooling = NetworkHelper::separateInStandaloneBranch(m.get_match_root(), defaultPrecisions);
    const bool updatePrecision = isPrecisionPreserved(pooling);
    const auto newOperation = moveDequantizationAfter(context, pooling, NetworkHelper::getDequantization(pooling, defaultPrecisions), updatePrecision);

    OPENVINO_DEBUG("LPT: done: ", newOperation);
    return true;
}

bool AvgPoolTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!LayerTransformation::canBeTransformed(context, operation)) {
        return false;
    }

    auto dequantization = NetworkHelper::getDequantization(operation, defaultPrecisions);

    return !dequantization.empty();
}

bool AvgPoolTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const {
    return NetworkHelper::isPrecisionPreserved(layer);
}

} // namespace low_precision
} // namespace pass
} // namespace ov
