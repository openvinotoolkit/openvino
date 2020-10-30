// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/fake_quantize_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "low_precision/common/dequantization_op.hpp"
#include "low_precision/network_helper.hpp"
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"

using namespace ngraph::pass::low_precision;

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::shared_ptr<ngraph::Function> FakeQuantizeFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fakeQuantizeOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
        input,
        element::f32,
        fakeQuantizeOnData.quantizationLevel,
        fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues,
        fakeQuantizeOnData.inputHighValues,
        fakeQuantizeOnData.outputLowValues,
        fakeQuantizeOnData.outputHighValues,
        "fakeQuantize");

    auto& rtInfo = fakeQuantize->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("fakeQuantize");

    const auto result = std::make_shared<ngraph::opset1::Result>(fakeQuantize);
    result->set_friendly_name("result");
    return std::make_shared<ngraph::Function>(result, ngraph::ParameterVector{ input }, "FakeQuantizeFunction");
}

std::shared_ptr<ngraph::Function> FakeQuantizeFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool updatePrecisions,
    const FakeQuantizeOnData& fakeQuantizeOnData,
    const ngraph::element::Type fakeQuantizeOutputPrecision,
    const DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    std::shared_ptr<ngraph::opset1::FakeQuantize> fakeQuantize = as_type_ptr<ngraph::opset1::FakeQuantize>(ngraph::builder::makeFakeQuantize(
        input,
        element::f32,
        fakeQuantizeOnData.quantizationLevel,
        fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues,
        fakeQuantizeOnData.inputHighValues,
        fakeQuantizeOnData.outputLowValues,
        fakeQuantizeOnData.outputHighValues,
        "fakeQuantize"));
    std::shared_ptr<Node> parent = fakeQuantize;

    auto& rtInfo = fakeQuantize->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("fakeQuantize");

    if (updatePrecisions) {
        parent = makeDequantization(parent, dequantization);
        ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize, fakeQuantizeOutputPrecision);
    } else {
        auto notUpdatePrecisionsDequantization = dequantization;
        if (fakeQuantize->get_output_element_type(0) != element::f32) {
            notUpdatePrecisionsDequantization.convert = { element::f32 };
        } else {
            notUpdatePrecisionsDequantization.convert = {};
        }
        parent = makeDequantization(parent, notUpdatePrecisionsDequantization);
    }

    parent->set_friendly_name("fakeQuantize");
    const auto result = std::make_shared<ngraph::opset1::Result>(parent);
    result->set_friendly_name("result");
    return std::make_shared<ngraph::Function>(result, ngraph::ParameterVector{ input }, "FakeQuantizeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
