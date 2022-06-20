// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/clamp.hpp"
#include <algorithm>
#include <memory>
#include <ngraph/ngraph.hpp>

#include <ngraph/pattern/op/wrap_type.hpp>
#include "low_precision/network_helper.hpp"
#include "itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

ClampTransformation::ClampTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(ClampTransformation);
    auto matcher = pattern::wrap_type<opset1::Clamp>({ pattern::wrap_type<opset1::Multiply>() });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        MATCHER_SCOPE_ENABLE(ClampTransformation);
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool ClampTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) {
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const std::shared_ptr<Node> clamp = NetworkHelper::separateInStandaloneBranch(m.get_match_root(), defaultPrecisions);
    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(clamp, defaultPrecisions);

    const bool moveSubtract = dequantization.subtract == nullptr ? false : NetworkHelper::isScalarLike(dequantization.subtractConstant);
    // issue #43136
    if (!moveSubtract && (dequantization.subtract != nullptr)) {
        return false;
    }

    const auto newClamp = ov::as_type_ptr<opset1::Clamp>(moveDequantizationAfter(context, clamp, dequantization, false, moveSubtract));

    std::shared_ptr<ngraph::opset1::Clamp> replacement;
    {
        double min = newClamp->get_min();
        double max = newClamp->get_max();

        if (dequantization.multiply != nullptr) {
            double scale = dequantization.multiplyConstant->cast_vector<double>()[0];
            if (scale < 0.0) {
                std::swap(min, max);
            }
            min /= scale;
            max /= scale;
        }

        if (dequantization.subtract != nullptr && moveSubtract) {
            double shift = dequantization.subtractConstant->cast_vector<double>()[0];
            min += shift;
            max += shift;
        }

        replacement = std::make_shared<ngraph::opset1::Clamp>(newClamp->input_value(0), min, max);
    }

    replace_node_update_name(newClamp, replacement);

    element::Type outputClampType = dequantization.multiply ?
        dequantization.multiply->get_output_element_type(0) :
        dequantization.subtract->get_output_element_type(0);
    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(replacement, outputClampType);
    return true;
}

bool ClampTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    if (!LayerTransformation::canBeTransformed(context, op)) {
        return false;
    }

    const auto dequantization = NetworkHelper::getDequantization(op, defaultPrecisions);
    if (dequantization.multiply == nullptr) {
        return false;
    }

    return NetworkHelper::isScalarLike(dequantization.multiplyConstant);
}

bool ClampTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
