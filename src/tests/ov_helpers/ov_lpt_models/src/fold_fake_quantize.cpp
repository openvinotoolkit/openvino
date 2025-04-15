// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/fold_fake_quantize.hpp"

#include "openvino/opsets/opset1.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace builder {
namespace subgraph {

using namespace ov::pass;

std::shared_ptr<ov::Model> FoldFakeQuantizeFunction::getOriginal(
    const ov::element::Type precision,
    const ov::Shape& constShape,
    const std::vector<float>& constValues,
    const FakeQuantizeOnData& fakeQuantizeOnData) {
    const auto constant = std::make_shared<ov::opset1::Constant>(precision, constShape, constValues);

    const auto fakeQuantize = ov::test::utils::make_fake_quantize(
        constant, precision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);
    fakeQuantize->set_friendly_name("fakeQuantize");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(fakeQuantize) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{}, "FoldFakeQuantizeFunction");
}

std::shared_ptr<ov::Model> FoldFakeQuantizeFunction::getReference(
    const ov::element::Type precision,
    const ov::Shape& constShape,
    const std::vector<float>& constValues) {
    const std::shared_ptr<Node> constant = std::make_shared<ov::opset1::Constant>(precision, constShape, constValues);

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(constant) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{}, "FoldFakeQuantizeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
