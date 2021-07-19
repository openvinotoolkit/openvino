// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/squeeze.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>

#include <ngraph/pattern/op/wrap_type.hpp>

#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::SqueezeTransformation, "SqueezeTransformation", 0);

SqueezeTransformation::SqueezeTransformation(const Params& params) : LayerTransformation(params) {
    auto matcher = pattern::wrap_type<opset1::Squeeze>({ pattern::wrap_type<opset1::Multiply>(), pattern::wrap_type<opset1::Constant>() });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "SqueezeTransformation");
    this->register_matcher(m, callback);
}

bool SqueezeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    auto squeezeOnConstant = [](const std::shared_ptr<ngraph::Node>& squeeze,
                                const std::shared_ptr<ngraph::opset1::Constant>& dequantizationOpConstant,
                                const ngraph::PartialShape& inputShape) {
        const size_t inputRankValue = inputShape.rank().get_length();
        if (dequantizationOpConstant->get_shape().size() == inputRankValue) {
            return as_type_ptr<opset1::Constant>(fold<opset1::Squeeze>(dequantizationOpConstant, squeeze->get_input_node_shared_ptr(1)));
        }
        return dequantizationOpConstant;
    };

    const std::shared_ptr<Node> squeeze = NetworkHelper::separateInStandaloneBranch(m.get_match_root());
    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(squeeze);

    if (dequantization.multiply != nullptr) {
        auto newConstant = squeezeOnConstant(squeeze, dequantization.multiplyConstant, dequantization.data.get_partial_shape());
        replace_node(dequantization.multiplyConstant, newConstant);
    }

    if (dequantization.subtract != nullptr) {
        auto newConstant = squeezeOnConstant(squeeze, dequantization.subtractConstant, dequantization.data.get_partial_shape());
        replace_node(dequantization.subtractConstant, newConstant);
    }

    moveDequantizationAfter(context, squeeze, NetworkHelper::getDequantization(squeeze), false);
    return true;
}

bool SqueezeTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

bool SqueezeTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    return (!NetworkHelper::getDequantization(layer).empty()) && LayerTransformation::canBeTransformed(context, layer);
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
