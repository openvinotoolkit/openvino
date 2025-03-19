// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/fake_quantize.hpp"

#include "openvino/opsets/opset1.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "low_precision/network_helper.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

using namespace ov::pass::low_precision;

namespace ov {
namespace builder {
namespace subgraph {

using namespace ov::pass;

std::shared_ptr<ov::Model> FakeQuantizeFunction::getOriginalWithMaxPool(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const FakeQuantizeOnData& fakeQuantizeOnData) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input->set_friendly_name("input");

    const auto fakeQuantize = ov::test::utils::make_fake_quantize(input,
                                                                  ov::element::f32,
                                                                  fakeQuantizeOnData.quantizationLevel,
                                                                  fakeQuantizeOnData.constantShape,
                                                                  fakeQuantizeOnData.inputLowValues,
                                                                  fakeQuantizeOnData.inputHighValues,
                                                                  fakeQuantizeOnData.outputLowValues,
                                                                  fakeQuantizeOnData.outputHighValues);
    const auto maxPool = std::make_shared<ov::opset1::MaxPool>(
        fakeQuantize,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 });

    fakeQuantize->set_friendly_name("fakeQuantize");
    auto& rtInfo = fakeQuantize->get_rt_info();
    rtInfo["Variant::std::string"] = "fakeQuantize";

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(maxPool) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "FakeQuantizeFunction");
}

std::shared_ptr<ov::Model> FakeQuantizeFunction::getOriginal(
    const ov::pass::low_precision::LayerTransformation::Params& params,
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const FakeQuantizeOnDataWithConstant& fakeQuantizeOnData,
    const bool addNotPrecisionPreservedOperation) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input->set_friendly_name("input");

    const auto fakeQuantize = makeFakeQuantize(input, ov::element::f32, fakeQuantizeOnData);
    fakeQuantize->set_friendly_name("fakeQuantize");
    auto& rtInfo = fakeQuantize->get_rt_info();
    rtInfo["Variant::std::string"] = "fakeQuantize";

    std::shared_ptr<Node> lastOperation = fakeQuantize;
    if (addNotPrecisionPreservedOperation) {
        lastOperation = std::make_shared<ov::opset1::AvgPool>(fakeQuantize,
                                                              Strides{1, 1},
                                                              Shape{1, 1},
                                                              Shape{1, 1},
                                                              Shape{2, 2},
                                                              true,
                                                              ov::op::RoundingType::FLOOR);
    }
    lastOperation->set_friendly_name("lastOperation");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(lastOperation) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "FakeQuantizeFunction");
}

std::shared_ptr<ov::Model> FakeQuantizeFunction::getReference(
    const ov::pass::low_precision::LayerTransformation::Params& params,
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const bool updatePrecisions,
    const FakeQuantizeOnDataWithConstant& fakeQuantizeOnData,
    const ov::element::Type fakeQuantizeOutputPrecision,
    const ov::builder::subgraph::DequantizationOperations& dequantization,
    const bool addNotPrecisionPreservedOperation) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input->set_friendly_name("input");

    auto fakeQuantize = makeFakeQuantizeTypeRelaxed(input, ov::element::f32, fakeQuantizeOnData);

    auto& rtInfo = fakeQuantize->get_rt_info();
    rtInfo["Variant::std::string"] = "fakeQuantize";

    std::shared_ptr<Node> lastOperation = fakeQuantize;
    if (addNotPrecisionPreservedOperation) {
        lastOperation = std::make_shared<ov::op::TypeRelaxed<ov::opset1::AvgPool>>(
            std::vector<element::Type>{element::f32},
            std::vector<element::Type>{element::f32},
            ov::op::TemporaryReplaceOutputType(fakeQuantize, element::f32).get(),
            Strides{1, 1},
            Shape{1, 1},
            Shape{1, 1},
            Shape{2, 2},
            true,
            op::RoundingType::FLOOR);
    }

    auto updateDequantization = dequantization;
    if (!updateDequantization.subtract.empty()) {
        updateDequantization.subtract.constantPrecision = ov::element::f32;
    }
    if (!updateDequantization.multiply.empty()) {
        updateDequantization.multiply.constantPrecision = ov::element::f32;
    }

    updateDequantization.multiply.outPrecision = precision;
    std::shared_ptr<Node> deq;
    if (updatePrecisions) {
        deq = makeDequantization(lastOperation, updateDequantization);
        ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize, fakeQuantizeOutputPrecision);
    } else {
        if (precision == ov::element::f32) {
            updateDequantization.convert = {};
        }
        deq = makeDequantization(lastOperation, updateDequantization);
        ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize, precision);
    }

    deq->set_friendly_name("lastOperation");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(deq) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "FakeQuantizeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
