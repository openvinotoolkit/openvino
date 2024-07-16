// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/squeeze.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/util/log.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "low_precision/network_helper.hpp"

namespace ov {
namespace pass {
namespace low_precision {

SqueezeTransformation::SqueezeTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(SqueezeTransformation);
    auto matcher = pattern::wrap_type<opset1::Squeeze>({ pattern::wrap_type<opset1::Multiply>(), pattern::wrap_type<opset1::Constant>() });

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

bool SqueezeTransformation::transform(TransformationContext& context, ov::pass::pattern::Matcher &m) {
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    auto squeezeOnConstant = [](const std::shared_ptr<ov::Node>& squeeze,
                                const std::shared_ptr<ov::opset1::Constant>& dequantizationOpConstant,
                                const ov::PartialShape& inputShape) {
        const size_t inputRankValue = inputShape.rank().get_length();
        const auto constantShape = dequantizationOpConstant->get_shape();
        if (shape_size(constantShape) == 1ul) {
            return NetworkHelper::toScalar(dequantizationOpConstant);
        }
        if (constantShape.size() == inputRankValue) {
            return ov::as_type_ptr<opset1::Constant>(fold<opset1::Squeeze>(dequantizationOpConstant, squeeze->input_value(1)));
        }

        return dequantizationOpConstant;
    };

    const std::shared_ptr<Node> squeeze = NetworkHelper::separateInStandaloneBranch(m.get_match_root(), defaultPrecisions);
    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(squeeze, defaultPrecisions);

    if (dequantization.multiply != nullptr) {
        auto newConstant = squeezeOnConstant(squeeze, dequantization.multiplyConstant, dequantization.data.get_partial_shape());
        replace_node(dequantization.multiplyConstant, newConstant);
    }

    if (dequantization.subtract != nullptr) {
        auto newConstant = squeezeOnConstant(squeeze, dequantization.subtractConstant, dequantization.data.get_partial_shape());
        replace_node(dequantization.subtractConstant, newConstant);
    }

    const auto newOperation = moveDequantizationAfter(context, squeeze, NetworkHelper::getDequantization(squeeze, defaultPrecisions));

    OPENVINO_DEBUG("LPT: done: ", newOperation);
    return true;
}

bool SqueezeTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

bool SqueezeTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    return (!NetworkHelper::getDequantization(layer, defaultPrecisions).empty()) && LayerTransformation::canBeTransformed(context, layer);
}

} // namespace low_precision
} // namespace pass
} // namespace ov
