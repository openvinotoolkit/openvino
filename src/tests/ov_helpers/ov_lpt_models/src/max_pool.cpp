// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/max_pool.hpp"

#include "openvino/opsets/opset1.hpp"
#include <ov_ops/type_relaxed.hpp>
#include "low_precision/network_helper.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> MaxPoolFunction::getOriginal(
    const ov::element::Type originalFunctionPrecision,
    const ov::PartialShape& inputShape,
    const FakeQuantizeOnData& fakeQuantizeOnData) {
    const auto input = std::make_shared<ov::opset1::Parameter>(originalFunctionPrecision, inputShape);

    const auto fakeQuantize = ov::test::utils::make_fake_quantize(
        input, originalFunctionPrecision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);

    const std::shared_ptr<ov::Node> maxPool = std::make_shared<ov::opset1::MaxPool>(fakeQuantize,
                                                                                    Strides{1, 1},
                                                                                    Shape{1, 1},
                                                                                    Shape{0, 0},
                                                                                    Shape{2, 2},
                                                                                    ov::op::RoundingType::FLOOR);

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(maxPool) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "MaxPoolTransformation");
}

std::shared_ptr<ov::Model> MaxPoolFunction::get(
    const ov::PartialShape& inputShape,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOperation,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);
    std::shared_ptr<ov::Node> parent = input;

    parent = makeDequantization(parent, dequantizationBefore);

    const auto maxPool = std::make_shared<ov::opset1::MaxPool>(parent,
                                                               Strides{1, 1},
                                                               Shape{1, 1},
                                                               Shape{0, 0},
                                                               Shape{2, 2},
                                                               ov::op::RoundingType::FLOOR);
    parent = maxPool;
    ov::pass::low_precision::NetworkHelper::setOutDataPrecision(maxPool, precisionAfterOperation);

    parent = makeDequantization(maxPool, dequantizationAfter);
    maxPool->set_friendly_name("maxPool");

    const std::shared_ptr<ov::opset1::Result> result = std::make_shared<ov::opset1::Result>(parent);

    const std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        ov::ResultVector{ result },
        std::vector<std::shared_ptr<ov::op::v0::Parameter>> { input },
        "MaxPoolTransformation");
    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
