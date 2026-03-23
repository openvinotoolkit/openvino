// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pad_string.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/pad.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace PadString {

std::string PadStringLayerCPUTest::getTestCaseName(
        const testing::TestParamInfo<PadStringLayerCPUTestParamsSet>& obj) {
    const auto& [basicParamsSet, cpuParams] = obj.param;
    const auto& [padStringParams, td] = basicParamsSet;
    const auto& [padsBegin, padsEnd, padValue, inputShape] = padStringParams;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::partialShape2str({inputShape.first}) << "_";
    result << "TS=(";
    for (const auto& shape : inputShape.second)
        result << ov::test::utils::vec2str(shape) << "_";
    result << ")";
    result << "_padsBegin=" << ov::test::utils::vec2str(padsBegin);
    result << "_padsEnd="   << ov::test::utils::vec2str(padsEnd);
    result << "_padValue="  << padValue;
    result << CPUTestsBase::getTestCaseName(cpuParams);
    return result.str();
}

void PadStringLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();

    const auto& dataShape = targetInputStaticShapes[0];
    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = 0;
    in_data.range = 10;
    inputs.insert({funcInputs[0].get_node_shared_ptr(),
                   ov::test::utils::create_and_fill_tensor(ov::element::string, dataShape, in_data)});
}

void PadStringLayerCPUTest::SetUp() {
    const auto& [basicParamsSet, cpuParams] = this->GetParam();
    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    const auto& [padStringParams, _targetDevice] = basicParamsSet;
    const auto& [padsBegin, padsEnd, padValue, inputShape] = padStringParams;
    targetDevice = _targetDevice;

    selectedType = "ref_string";

    init_input_shapes({inputShape});

    auto dataParam = std::make_shared<ov::op::v0::Parameter>(ov::element::string, inputDynamicShapes[0]);
    auto padsBeginConst = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{padsBegin.size()}, padsBegin.data());
    auto padsEndConst = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{padsEnd.size()}, padsEnd.data());
    auto padValueConst = std::make_shared<ov::op::v0::Constant>(
        ov::element::string, ov::Shape{}, std::vector<std::string>{padValue});

    auto pad = std::make_shared<ov::op::v12::Pad>(
        dataParam, padsBeginConst, padsEndConst, padValueConst, ov::op::PadMode::CONSTANT);

    function = std::make_shared<ov::Model>(pad->outputs(), ov::ParameterVector{dataParam});
}

TEST_P(PadStringLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Pad");
}

const std::vector<PadStringSpecificParams> PadStringParamsVector = {
    // 1-D: pad only at begin
    PadStringSpecificParams{{2}, {0}, "<pad>", InputShape{{}, {{4}}}},
    // 1-D: pad only at end
    PadStringSpecificParams{{0}, {3}, "", InputShape{{}, {{3}}}},
    // 1-D: pad both sides with non-empty pad value
    PadStringSpecificParams{{1}, {2}, "FILL", InputShape{{}, {{5}}}},
    // 2-D: pad rows and columns
    PadStringSpecificParams{{1, 1}, {1, 1}, "_", InputShape{{}, {{3, 4}}}},
    // 2-D: pad only rows
    PadStringSpecificParams{{2, 0}, {1, 0}, "X", InputShape{{}, {{2, 3}}}},
    // 2-D: pad only columns
    PadStringSpecificParams{{0, 1}, {0, 2}, "", InputShape{{}, {{2, 3}}}},
    // negative (crop) on begin
    PadStringSpecificParams{{-1}, {0}, "", InputShape{{}, {{4}}}},
    // dynamic shape, 1-D
    PadStringSpecificParams{{1}, {1}, "<pad>", InputShape{{-1}, {{3}, {5}}}},
    // dynamic shape, 2-D
    PadStringSpecificParams{{1, 1}, {1, 1}, "_", InputShape{{-1, -1}, {{2, 3}, {4, 2}}}},
    // empty pad value
    PadStringSpecificParams{{0}, {2}, "", InputShape{{}, {{3}}}},
    // 3-D
    PadStringSpecificParams{{1, 0, 0}, {0, 1, 0}, "?", InputShape{{}, {{2, 3, 4}}}},
};

}  // namespace PadString
}  // namespace test
}  // namespace ov
