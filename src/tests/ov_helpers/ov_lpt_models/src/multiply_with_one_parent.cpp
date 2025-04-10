// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/multiply_with_one_parent.hpp"

#include "openvino/opsets/opset1.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> MultiplyWithOneParentFunction::getOriginal(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const FakeQuantizeOnData& fqOnData) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);

    const auto fakeQuantize = ov::test::utils::make_fake_quantize(
            input, precision, fqOnData.quantizationLevel, fqOnData.constantShape,
        fqOnData.inputLowValues, fqOnData.inputHighValues, fqOnData.outputLowValues, fqOnData.outputHighValues);

    const auto multiply = std::make_shared<ov::opset1::Multiply>(fakeQuantize->output(0), fakeQuantize->output(0));

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(multiply) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "MultiplyWithOneParentFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
