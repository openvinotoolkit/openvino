// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/shuffle_channels.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {
ShuffleChannelsTransformation::ShuffleChannelsTransformation(const Params& params) : LayerTransformation(params) {}

void ShuffleChannelsTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::ShuffleChannels>({ make_op_label<opset1::Multiply>() }));
}

bool ShuffleChannelsTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) const {
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
