// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/fuse_fake_quantize_and_scale_shift.hpp"

#include "openvino/opsets/opset1.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace builder {
namespace subgraph {

using namespace ov::pass;

std::shared_ptr<ov::Model> FuseFakeQuantizeAndScaleShiftFunction::getOriginal(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const FakeQuantizeOnData& fakeQuantizeOnData) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input->set_friendly_name("input");

    const std::shared_ptr<Node> fakeQuantize = ov::test::utils::make_fake_quantize(
        input, precision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);
    fakeQuantize->set_friendly_name("fakeQuantize");

    const std::shared_ptr<Node> multiply = std::make_shared<ov::opset1::Multiply>(
        fakeQuantize,
        std::make_shared<ov::opset1::Constant>(precision, Shape{ 1, 1, 1, 1 }, std::vector<float>({ 150 })));

    const std::shared_ptr<Node> add = std::make_shared<ov::opset1::Add>(
        multiply,
        std::make_shared<ov::opset1::Constant>(precision, Shape{ 1, 1, 1, 1 }, std::vector<float>({ 127.5 })));
    add->set_friendly_name("output");

    const ov::ResultVector results{ std::make_shared<ov::opset1::Result>(add) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "FuseFakeQuantizeAndScaleShiftFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
