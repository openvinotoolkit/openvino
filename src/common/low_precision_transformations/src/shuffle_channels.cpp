// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/shuffle_channels.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>

#include <ngraph/pattern/op/wrap_type.hpp>

#include "low_precision/network_helper.hpp"
#include "itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

ShuffleChannelsTransformation::ShuffleChannelsTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(ShuffleChannelsTransformation);
    auto matcher = pattern::wrap_type<opset1::ShuffleChannels>({ pattern::wrap_type<opset1::Multiply>() });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        MATCHER_SCOPE_ENABLE(ShuffleChannelsTransformation);
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool ShuffleChannelsTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) {
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const auto shuffleChannels = ov::as_type_ptr<opset1::ShuffleChannels>(NetworkHelper::separateInStandaloneBranch(m.get_match_root(), defaultPrecisions));
    auto dequantization = NetworkHelper::getDequantization(shuffleChannels, defaultPrecisions);

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
                return ov::as_type_ptr<opset1::Constant>(shuffledConst);
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

    const auto shuffleChannels = ov::as_type_ptr<opset1::ShuffleChannels>(op);
    if (shuffleChannels == nullptr) {
        return false;
    }

    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(shuffleChannels, defaultPrecisions);
    if (dequantization.empty()) {
        return false;
    }

    // It's impossible to normalize a negative axis in case of dynamic rank
    // but it's necessary when dequantization operations are per channel
    if (shuffleChannels->get_input_partial_shape(0).rank().is_dynamic() && shuffleChannels->get_axis() < 0) {
        const bool perChannelSub = dequantization.subtractConstant ?
            ov::shape_size(dequantization.subtractConstant->get_shape()) > 0 :
            false;
        const bool perChannelMul = dequantization.multiplyConstant ?
            ov::shape_size(dequantization.multiplyConstant->get_shape()) > 0 :
            false;
        if (perChannelMul || perChannelSub) {
            return false;
        }
    }

    return true;
}

bool ShuffleChannelsTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
