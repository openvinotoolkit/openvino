// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/unsqueeze.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

UnsqueezeTransformation::UnsqueezeTransformation(const Params& params) : LayerTransformation(params) {
}

void UnsqueezeTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Unsqueeze>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::Constant>() }));
}

void UnsqueezeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    if (!LayerTransformation::canBeTransformed(context, m.get_match_root())) {
        return;
    }

    auto unsqueezeOnConstant = [](const std::shared_ptr<ngraph::Node>& unsqueeze,
                                const std::shared_ptr<ngraph::Node>& dequantizationOperation,
                                const ngraph::Shape& inputShape) {
        std::shared_ptr<ngraph::Node> dequantizationOpConstant = dequantizationOperation->get_argument(1);
        if (dequantizationOpConstant->get_shape() == inputShape && dequantizationOpConstant->get_shape().size() > 1) {
            return fold<opset1::Unsqueeze>(dequantizationOpConstant, unsqueeze->get_argument(1));
        }
        return dequantizationOpConstant;
    };

    const std::shared_ptr<Node> unsqueeze = separateInStandaloneBranch(m.get_match_root());
    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(unsqueeze);

    if (dequantization.multiply != nullptr) {
        auto newConstant = unsqueezeOnConstant(unsqueeze, dequantization.multiply, dequantization.data->get_shape());
        dequantization.multiply->set_argument(1, newConstant);
    }

    if (dequantization.subtract != nullptr) {
        auto newConstant = unsqueezeOnConstant(unsqueeze, dequantization.subtract, dequantization.data->get_shape());
        dequantization.subtract->set_argument(1, newConstant);
    }

    moveDequantizationAfter(context, unsqueeze, dequantization, false);
}


} // namespace low_precision
} // namespace pass
} // namespace ngraph
