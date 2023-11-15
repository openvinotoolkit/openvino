// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset13.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include "scaled_attn.hpp"
#include "gtest/gtest.h"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

std::string ScaledAttnLayerCPUTest::getTestCaseName(const testing::TestParamInfo<ScaledAttnCPUTestParams>& obj) {
    CPUSpecificParams cpuParams;
    ElementType inType;
    ScaledAttnConfig config;
    std::string targetDevice;
    std::tie(inType, config, targetDevice, cpuParams) = obj.param;

    std::ostringstream result;
    result << "netPRC=" << inType << "_";
    result << "IS=" << ov::test::utils::partialShape2str({config.inputShape.first}) << "_";
    result << "TS=";
    for (const auto& shape : config.inputShape.second) {
        result << "(";
        result << ov::test::utils::vec2str(shape);
        result << ")_";
    }
    result << "is_causal=" << config.is_causal << "_";
    result << "has_attn=" << config.has_attn << "_";
    result << "has_scale=" << config.has_scale << "_";
    result << "trgDev=" << targetDevice;
    result << CPUTestsBase::getTestCaseName(cpuParams);

    return result.str();
}

void ScaledAttnLayerCPUTest::SetUp() {
    ElementType inType;
    CPUSpecificParams cpuParams;
    std::tie(inType, config, targetDevice, cpuParams) = this->GetParam();

    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    if (selectedType.empty()) {
        selectedType = getPrimitiveType();
    }

    if (inType == ElementType::bf16) {
        rel_threshold = 2e-2f;
    }
    selectedType = makeSelectedTypeStr(selectedType, inType);
    init_input_shapes({config.inputShape});
    ov::ParameterVector inputParams;
    // q,k,v
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]));
    inputParams[0]->set_friendly_name("q");
    inputParams[1]->set_friendly_name("k");
    inputParams[2]->set_friendly_name("v");
    if (config.has_attn) {
        // attention_mask
        // shape: [B, L0+L1]
        ov::PartialShape attnShape{inputDynamicShapes[0][0], -1};
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, attnShape));
        inputParams.back()->set_friendly_name("attention_mask");
    }
    if (config.has_scale) {
        // scale
        // shape: [1]
        ASSERT_TRUE(config.has_attn);       // must have attention_mask input
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::PartialShape{1}));
        inputParams.back()->set_friendly_name("scale");
    }
    ov::OutputVector inputs;
    for (auto& input : inputParams) {
        inputs.push_back(input);
    }
    auto sdp = std::make_shared<ov::opset13::ScaledDotProductAttention>(inputs, config.is_causal);
    sdp->set_friendly_name("mha");
    function = makeNgraphFunction(inType, inputParams, sdp, "SDP");

    functionRefs = function->clone();
    ov::pass::Manager manager;
    // decompose ScaledDotProductAttention
    manager.register_pass<ov::pass::CommonOptimizations>();
    manager.run_passes(functionRefs);
}

void ScaledAttnLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    std::vector<ov::Shape> shapes(3);
    shapes[0] = targetInputStaticShapes[0];
    shapes[1] = targetInputStaticShapes[0];
    shapes[2] = targetInputStaticShapes[0];
    if (config.has_attn) {
        shapes.push_back(ov::Shape{targetInputStaticShapes[0][0], targetInputStaticShapes[0][2]});
    }
    if (config.has_scale) {
        shapes.push_back(ov::Shape{1});
    }
    SubgraphBaseTest::generate_inputs(shapes);
}

TEST_P(ScaledAttnLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "ScaledAttn");
}

namespace ScaledAttn {

}  // namespace ScaledAttn
}  // namespace CPULayerTestsDefinitions
