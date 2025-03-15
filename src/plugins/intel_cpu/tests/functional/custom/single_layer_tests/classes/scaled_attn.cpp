// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scaled_attn.hpp"

#include "gtest/gtest.h"
#include "openvino/opsets/opset13.hpp"
#include "utils/cpu_test_utils.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"
#include "openvino/pass/manager.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

std::string ScaledAttnLayerCPUTest::getTestCaseName(const testing::TestParamInfo<ScaledAttnCPUTestParams>& obj) {
    CPUSpecificParams cpuParams;
    ElementType inType;
    std::vector<InputShape> inputShapes;
    bool is_causal;
    bool has_attn;
    bool has_scale;
    std::string targetDevice;
    std::tie(inType, inputShapes, is_causal, has_attn, has_scale, targetDevice, cpuParams) = obj.param;

    std::ostringstream result;
    result << "netPRC=" << inType << "_";
    result << "IS=";
    for (const auto& inputShape : inputShapes) {
        result << ov::test::utils::partialShape2str({inputShape.first}) << "_";
    }
    result << "TS=";
    for (const auto& shapes : inputShapes) {
        for (const auto& shape : shapes.second) {
            result << ov::test::utils::vec2str(shape);
            result << "_";
        }
    }
    result << "is_causal=" << is_causal << "_";
    result << "has_attn=" << has_attn << "_";
    result << "has_scale=" << has_scale << "_";
    result << "trgDev=" << targetDevice;
    result << CPUTestsBase::getTestCaseName(cpuParams);

    return result.str();
}

void ScaledAttnLayerCPUTest::SetUp() {
    ElementType inType;
    CPUSpecificParams cpuParams;
    std::vector<InputShape> inputShapes;
    std::tie(inType, inputShapes, is_causal, has_attn, has_scale, targetDevice, cpuParams) = this->GetParam();

    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    if (selectedType.empty()) {
        selectedType = getPrimitiveType();
    }

    if (inType == ElementType::bf16) {
        // Issue: 163144
        if (with_cpu_x86_avx2_vnni_2()) {
            rel_threshold = 0.2f;
        } else {
            rel_threshold = 2e-2f;
        }
    }
    selectedType = makeSelectedTypeStr(selectedType, inType);
    init_input_shapes(inputShapes);
    ov::ParameterVector inputParams;
    // q,k,v
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[1]));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[1]));
    inputParams[0]->set_friendly_name("q");
    inputParams[1]->set_friendly_name("k");
    inputParams[2]->set_friendly_name("v");
    // special case: only scale but no attn
    if (!has_attn && has_scale) {
        // attention_mask：[1]
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::PartialShape{}));
        inputParams.back()->set_friendly_name("attention_mask");
        // scale：[1]
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::PartialShape{1}));
        inputParams.back()->set_friendly_name("scale");
    } else {
        if (has_attn) {
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[2]));
            inputParams.back()->set_friendly_name("attention_mask");
        }
        if (has_scale) {
            // scale：[1]
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::PartialShape{1}));
            inputParams.back()->set_friendly_name("scale");
        }
    }
    ov::OutputVector inputs;
    for (auto& input : inputParams) {
        inputs.push_back(input);
    }
    auto sdp = std::make_shared<ov::opset13::ScaledDotProductAttention>(inputs, is_causal);
    sdp->set_friendly_name("mha");
    function = makeNgraphFunction(inType, inputParams, sdp, "SDP");

    functionRefs = function->clone();
    ov::pass::Manager manager;
    // decompose ScaledDotProductAttention
    manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
    manager.run_passes(functionRefs);
}

void ScaledAttnLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    std::vector<ov::Shape> shapes(3);
    shapes[0] = targetInputStaticShapes[0];
    shapes[1] = targetInputStaticShapes[1];
    shapes[2] = targetInputStaticShapes[1];
    if (!has_attn && has_scale) {
        shapes.push_back(ov::Shape{});
        shapes.push_back(ov::Shape{1});
    } else {
        if (has_attn) {
            shapes.push_back(targetInputStaticShapes[2]);
        }
        if (has_scale) {
            shapes.push_back(ov::Shape{1});
        }
    }
    SubgraphBaseTest::generate_inputs(shapes);
}

TEST_P(ScaledAttnLayerCPUTest, CompareWithRefs) {
    CPUSpecificParams cpuParams;
    ElementType inType;
    std::vector<InputShape> inputShapes;
    bool is_causal;
    bool has_attn;
    bool has_scale;
    std::string targetDevice;
    std::tie(inType, inputShapes, is_causal, has_attn, has_scale, targetDevice, cpuParams) = this->GetParam();
    if (inType == ElementType::bf16 && !ov::with_cpu_x86_bfloat16() && !with_cpu_x86_avx2_vnni_2())
        GTEST_SKIP();
    run();
    CheckPluginRelatedResults(compiledModel, "ScaledAttn");
}

namespace ScaledAttn {

}  // namespace ScaledAttn
}  // namespace test
}  // namespace ov
