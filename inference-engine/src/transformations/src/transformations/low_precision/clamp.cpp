// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/clamp.hpp"
#include <algorithm>
#include <memory>
#include <ngraph/ngraph.hpp>
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

ClampTransformation::ClampTransformation(const Params& params) : LayerTransformation(params) {}

void ClampTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(pass,
               context,
               make_op_pattern<opset1::Clamp>({ make_op_label<opset1::Multiply>() }));
}

void ClampTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) const {
    if (!canBeTransformed(context, m.get_match_root())) {
        return;
    }

    const std::shared_ptr<Node> clamp = separateInStandaloneBranch(m.get_match_root());
    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(clamp);

    const std::shared_ptr<opset1::Clamp> newClamp = as_type_ptr<opset1::Clamp>(moveDequantizationAfter(context, clamp, dequantization, false));
    double min = newClamp->get_min();
    double max = newClamp->get_max();

    if (dequantization.multiply != nullptr) {
        std::vector<double> scale = as_type_ptr<opset1::Constant>(dequantization.multiply->get_input_node_shared_ptr(1))->cast_vector<double>();
        if (scale[0] < 0.0) {
            std::swap(min, max);
        }
        min /= scale[0];
        max /= scale[0];
    }

    if (dequantization.subtract != nullptr) {
        std::vector<double> shift = as_type_ptr<opset1::Constant>(dequantization.subtract->get_input_node_shared_ptr(1))->cast_vector<double>();
        min += shift[0];
        max += shift[0];
    }

    const std::shared_ptr<ngraph::opset1::Clamp> replacement = std::make_shared<ngraph::opset1::Clamp>(newClamp->get_input_node_shared_ptr(0), min, max);
    replace_node(newClamp, replacement);

    element::Type outputClampType = dequantization.multiply ?
        dequantization.multiply->get_output_element_type(0) :
        dequantization.subtract->get_output_element_type(0);
    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(replacement, outputClampType);
}

bool ClampTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    if (!LayerTransformation::canBeTransformed(context, op)) {
        return false;
    }
    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(op);

    Shape subtractShape = dequantization.subtract ? dequantization.subtract->get_input_shape(1) : Shape{ };
    if (!subtractShape.empty() && std::any_of(subtractShape.cbegin(), subtractShape.cend(), [](const size_t value) { return value != 1U; })) {
        return false;
    }

    Shape multiplyShape = dequantization.multiply ? dequantization.multiply->get_input_shape(1) : Shape{ };
    if (!multiplyShape.empty() && std::any_of(multiplyShape.cbegin(), multiplyShape.cend(), [](const size_t value) { return value != 1U; })) {
        return false;
    }
    return true;
}

bool ClampTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
