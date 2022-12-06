// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/gather.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"
#include "itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

GatherTransformation::GatherTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(GatherTransformation);
    auto matcher = pattern::wrap_type<opset8::Gather>({ pattern::wrap_type<opset1::Multiply>(),
                                                        pattern::wrap_type<ngraph::opset1::Constant>(),
                                                        pattern::wrap_type<ngraph::opset1::Constant>() });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool GatherTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    auto node = m.get_match_root();
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const std::shared_ptr<Node> gather = NetworkHelper::separateInStandaloneBranch(m.get_match_root(), defaultPrecisions);
    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(gather, defaultPrecisions);

    if (dequantization.multiply != nullptr) {
        auto newConstant = NetworkHelper::toScalar(dequantization.multiplyConstant);
        replace_node(dequantization.multiplyConstant, newConstant);
    }
    if (dequantization.subtract != nullptr) {
        auto newConstant = NetworkHelper::toScalar(dequantization.subtractConstant);
        replace_node(dequantization.subtractConstant, newConstant);
    }

    moveDequantizationAfter(context, gather, NetworkHelper::getDequantization(gather, defaultPrecisions), false);
    return true;
}

bool GatherTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!LayerTransformation::canBeTransformed(context, operation)) {
        return false;
    }

    auto dequantization = NetworkHelper::getDequantization(operation, defaultPrecisions);
    if (dequantization.empty()) {
        return false;
    }

    if (dequantization.multiply != nullptr) {
        if (!NetworkHelper::isScalarLike(dequantization.multiplyConstant)) {
            return false;
        }
    }
    if (dequantization.subtract != nullptr) {
        if (!NetworkHelper::isScalarLike(dequantization.subtractConstant)) {
            return false;
        }
    }
    return true;
}

bool GatherTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const {
    return NetworkHelper::isPrecisionPreserved(layer);
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
