// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/mat_mul.hpp"

#include <memory>
#include <string>
#include <vector>

#include "transformations/low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::pass::low_precision;

void MatMulTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<ngraph::Node> matMul = m.get_match_root();
    if (!canBeTransformed(context, matMul)) {
        return;
    }

    matMul = separateInStandaloneBranch(matMul);

    FakeQuantizeDequantization dequantization2 = ngraph::pass::low_precision::NetworkHelper::getDequantization(matMul, 1);
    if (dequantization2.empty()) {
        const std::shared_ptr<opset1::FakeQuantize> fakeQuantize = as_type_ptr<opset1::FakeQuantize>(dequantization2.data);
        if (fakeQuantize != nullptr) {
            const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fakeQuantize);
            const DataPrecision dataPrecision = getDataPrecision(fakeQuantize, quantizationDetails, true, supportAsymmetricQuantization);

            auto tuple = NetworkHelper::decomposeFakeQuantize(
                fakeQuantize,
                dataPrecision.precision,
                dataPrecision.min,
                dataPrecision.max,
                dataPrecision.hasZeroPoint,
                updatePrecisions);

            dequantization2 = ngraph::pass::low_precision::NetworkHelper::getDequantization(matMul, 1);
        }
    }

    const FakeQuantizeDequantization dequantization1 = ngraph::pass::low_precision::NetworkHelper::getDequantization(matMul, 0);
    const std::shared_ptr<opset1::MatMul> newMatMul = std::make_shared<ngraph::op::TypeRelaxed<opset1::MatMul>>(dequantization1.data, dequantization2.data);
    newMatMul->set_output_type(0, matMul->get_output_element_type(0), matMul->get_output_partial_shape(0));

    const std::shared_ptr<opset1::Multiply> newMultiply = std::make_shared<opset1::Multiply>(
        newMatMul,
        NetworkHelper::toScalarIfPossible(fold<ngraph::opset1::Multiply>(
            dequantization1.multiply->get_input_node_shared_ptr(1),
            dequantization2.multiply->get_input_node_shared_ptr(1))));
    replace_node(matMul, newMultiply);

    updateOutput(context, newMultiply, matMul);
}

void MatMulTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::MatMul>({ make_op_label<ngraph::opset1::Multiply>(), make_op_label<ngraph::opset1::Multiply>() }));

    addPattern(
        pass,
        context,
        make_op_pattern<opset1::MatMul>({ make_op_label<ngraph::opset1::Multiply>(), make_op_label<ngraph::opset1::FakeQuantize>() }));
}

bool MatMulTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

bool MatMulTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    if (!LayerTransformation::canBeTransformed(context, layer)) {
        return false;
    }

    return true;
}
