// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/avg_pool.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"
#include "itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

AvgPoolTransformation::AvgPoolTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(AvgPoolTransformation);
    auto matcher = pattern::wrap_type<opset1::AvgPool>({ pattern::wrap_type<opset1::Multiply>() });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        MATCHER_SCOPE_ENABLE(AvgPoolTransformation);
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool AvgPoolTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const std::shared_ptr<Node> pooling = NetworkHelper::separateInStandaloneBranch(m.get_match_root(), defaultPrecisions);
    const bool updatePrecision = isPrecisionPreserved(pooling);
    moveDequantizationAfter(context, pooling, NetworkHelper::getDequantization(pooling, defaultPrecisions), updatePrecision);
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
} // namespace ngraph
