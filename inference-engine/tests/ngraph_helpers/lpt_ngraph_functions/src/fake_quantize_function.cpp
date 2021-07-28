// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/fake_quantize_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "low_precision/common/dequantization_op.hpp"
#include "low_precision/network_helper.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"


using namespace ngraph::pass::low_precision;

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::shared_ptr<ngraph::Function> FakeQuantizeFunction::getOriginalWithMaxPool(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const FakeQuantizeOnData& fakeQuantizeOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input->set_friendly_name("input");

    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
        input, element::f32, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);
    const auto maxPool = std::make_shared<opset1::MaxPool>(
        fakeQuantize,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 });

    fakeQuantize->set_friendly_name("fakeQuantize");
    auto& rtInfo = fakeQuantize->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("fakeQuantize");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(maxPool) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FakeQuantizeFunction");
}

std::shared_ptr<ngraph::Function> FakeQuantizeFunction::getOriginal(
    const ngraph::pass::low_precision::LayerTransformation::Params& params,
    const ngraph::element::Type precision,
    const ngraph::PartialShape& inputShape,
    const FakeQuantizeOnDataWithConstant& fakeQuantizeOnData,
    const bool addNotPrecisionPreservedOperation) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input->set_friendly_name("input");

    const auto fakeQuantize = makeFakeQuantize(input, ngraph::element::f32, fakeQuantizeOnData);
    fakeQuantize->set_friendly_name("fakeQuantize");
    auto& rtInfo = fakeQuantize->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("fakeQuantize");

    std::shared_ptr<Node> lastOperation = fakeQuantize;
    if (addNotPrecisionPreservedOperation) {
        lastOperation = std::make_shared<opset1::AvgPool>(
            fakeQuantize,
            Strides{ 1, 1 },
            Shape{ 1, 1 },
            Shape{ 1, 1 },
            Shape{ 2, 2 },
            true,
            op::RoundingType::FLOOR);
    }
    lastOperation->set_friendly_name("lastOperation");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(lastOperation) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FakeQuantizeFunction");
}

std::shared_ptr<ngraph::Function> FakeQuantizeFunction::getReference(
    const ngraph::pass::low_precision::LayerTransformation::Params& params,
    const ngraph::element::Type precision,
    const ngraph::PartialShape& inputShape,
    const bool updatePrecisions,
    const FakeQuantizeOnDataWithConstant& fakeQuantizeOnData,
    const ngraph::element::Type fakeQuantizeOutputPrecision,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization,
    const bool addNotPrecisionPreservedOperation) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input->set_friendly_name("input");

    auto fakeQuantize = makeFakeQuantizeTypeRelaxed(input, ngraph::element::f32, fakeQuantizeOnData);

    auto& rtInfo = fakeQuantize->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("fakeQuantize");

    std::shared_ptr<Node> lastOperation = fakeQuantize;
    if (addNotPrecisionPreservedOperation) {
        lastOperation = std::make_shared<op::TypeRelaxed<opset1::AvgPool>>(
            std::vector<element::Type>{element::f32}, std::vector<element::Type>{element::f32},
            ngraph::op::TemporaryReplaceOutputType(fakeQuantize, element::f32).get(),
            Strides{ 1, 1 },
            Shape{ 1, 1 },
            Shape{ 1, 1 },
            Shape{ 2, 2 },
            true,
            op::RoundingType::FLOOR);
    }

    auto updateDequantization = dequantization;
    if (!updateDequantization.subtract.empty()) {
        updateDequantization.subtract.constantPrecision = element::f32;
    }
    if (!updateDequantization.multiply.empty()) {
        updateDequantization.multiply.constantPrecision = element::f32;
    }

    updateDequantization.multiply.outPrecision = precision;
    std::shared_ptr<Node> deq;
    if (updatePrecisions) {
        deq = makeDequantization(lastOperation, updateDequantization);
        ngraph::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize, fakeQuantizeOutputPrecision);
    } else {
        if (precision == element::f32) {
            updateDequantization.convert = {};
        }
        deq = makeDequantization(lastOperation, updateDequantization);
        ngraph::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize, precision);
    }

    deq->set_friendly_name("lastOperation");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(deq) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FakeQuantizeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
