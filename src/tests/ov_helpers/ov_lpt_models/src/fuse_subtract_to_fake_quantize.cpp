// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/fuse_subtract_to_fake_quantize.hpp"

#include "openvino/opsets/opset1.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "low_precision/network_helper.hpp"

#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ov {
namespace builder {
namespace subgraph {

using namespace ov::pass;

std::shared_ptr<ov::Model> FuseSubtractToFakeQuantizeFunction::get(
    const ov::PartialShape& inputShape,
    const FakeQuantizeOnDataWithConstant& fqOnData,
    const DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, inputShape);

    const auto constantPrecision = fqOnData.constantPrecision != ov::element::dynamic ? fqOnData.constantPrecision : ov::element::f32;
    const auto fakeQuantize = makeFakeQuantize(input, constantPrecision, fqOnData);
    const auto lastDequantization = makeDequantization(fakeQuantize, dequantization);
    lastDequantization->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(lastDequantization) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "FuseSubtractToFakeQuantizeFunction");
}

std::shared_ptr<ov::Model> FuseSubtractToFakeQuantizeFunction::get(
    const ov::PartialShape& inputShape,
    const FakeQuantizeOnDataWithConstant& fqOnData,
    const DequantizationOperations& dequantization,
    const FakeQuantizeOnDataWithConstant& fqOnData2,
    const DequantizationOperations& dequantization2) {
    const auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, inputShape);

    const auto axis = std::make_shared<ov::opset1::Constant>(ov::element::i64, Shape{}, 1ul);
    const std::shared_ptr<Node> split = std::make_shared<ov::opset1::Split>(input, axis, 2ul);

    const auto fakeQuantize = makeFakeQuantize(split->output(0), ov::element::f32, fqOnData);
    fakeQuantize->set_friendly_name("fakeQuantize");
    const auto lastDequantization = makeDequantization(fakeQuantize, dequantization);
    lastDequantization->set_friendly_name("output");

    const auto fakeQuantize2 = makeFakeQuantize(split->output(1), ov::element::f32, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");
    const auto lastDequantization2 = makeDequantization(fakeQuantize2, dequantization);
    lastDequantization2->set_friendly_name("output2");

    ov::ResultVector results{
        std::make_shared<ov::opset1::Result>(lastDequantization),
        std::make_shared<ov::opset1::Result>(lastDequantization2)
    };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "FuseSubtractToFakeQuantizeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
