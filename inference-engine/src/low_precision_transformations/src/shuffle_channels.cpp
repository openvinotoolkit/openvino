// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/shuffle_channels.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>

#include <ngraph/pattern/op/wrap_type.hpp>

#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::ShuffleChannelsTransformation, "ShuffleChannelsTransformation", 0);

ShuffleChannelsTransformation::ShuffleChannelsTransformation(const Params& params) : LayerTransformation(params) {
    auto matcher = pattern::wrap_type<opset1::ShuffleChannels>({ pattern::wrap_type<opset1::Multiply>() });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "ShuffleChannelsTransformation");
    this->register_matcher(m, callback);
}

bool ShuffleChannelsTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) {
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const auto shuffleChannels = as_type_ptr<opset1::ShuffleChannels>(NetworkHelper::separateInStandaloneBranch(m.get_match_root()));
    auto dequantization = NetworkHelper::getDequantization(shuffleChannels);

    const auto shuffleDequantizationConstant = [&](const std::shared_ptr<Node>& eltwise) {
        const auto normalizedConst = NetworkHelper::normalizeDequantizationShape(eltwise);
        const auto constShape = normalizedConst->get_shape();

        if (shape_size(constShape) == 1ul) {
            return NetworkHelper::toScalar(normalizedConst);
        } else {
            const size_t normalizedAxis = ngraph::normalize_axis(
                shuffleChannels->get_friendly_name(),
                shuffleChannels->get_axis(),
                shuffleChannels->get_input_partial_shape(0).rank());

            if (constShape[normalizedAxis] == 1ul) {
                return normalizedConst;
            } else {
                const auto group = shuffleChannels->get_group();
                const auto shuffledConst = fold<ngraph::opset1::ShuffleChannels>(normalizedConst, normalizedAxis, group);
                return as_type_ptr<opset1::Constant>(shuffledConst);
            }
        }
    };

    if (dequantization.subtract) {
        const auto shuffledSubConst = shuffleDequantizationConstant(dequantization.subtract);
        replace_node(dequantization.subtractConstant, shuffledSubConst);
        dequantization.subtractConstant = shuffledSubConst;
    }

    const auto shuffledMulConst = shuffleDequantizationConstant(dequantization.multiply);
    replace_node(dequantization.multiplyConstant, shuffledMulConst);
    dequantization.multiplyConstant = shuffledMulConst;

    moveDequantizationAfter(context, shuffleChannels, dequantization, false);
    return true;
}

bool ShuffleChannelsTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    if (!LayerTransformation::canBeTransformedSpatialDimension(context, op)) {
        return false;
    }

    const auto shuffleChannels = as_type_ptr<opset1::ShuffleChannels>(op);
    if (shuffleChannels == nullptr) {
        return false;
    }

    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(shuffleChannels);
    if (dequantization.empty()) {
        return false;
    }

    return true;
}

bool ShuffleChannelsTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
