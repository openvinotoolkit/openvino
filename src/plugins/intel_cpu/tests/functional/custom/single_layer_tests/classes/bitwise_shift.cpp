// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bitwise_shift.hpp"

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "gtest/gtest.h"
#include "internal_properties.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/properties.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
using ov::test::utils::EltwiseTypes;
using ov::test::utils::InputLayerType;
using ov::test::utils::OpType;

std::string BitwiseShiftLayerCPUTest::getTestCaseName(testing::TestParamInfo<BitshiftLayerCPUTestParamsSet> obj) {
    EltwiseTestParams basicParamsSet;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    bool enforceSnippets;
    ov::AnyMap val_map;
    std::tie(basicParamsSet, cpuParams, fusingParams, enforceSnippets, val_map) = obj.param;

    std::ostringstream result;
    result << EltwiseLayerTest::getTestCaseName(testing::TestParamInfo<EltwiseTestParams>(basicParamsSet, 0));
    result << CPUTestsBase::getTestCaseName(cpuParams);
    result << CpuTestWithFusing::getTestCaseName(fusingParams);
    result << "_enforceSnippets=" << enforceSnippets;

    return result.str();
}

ov::Tensor BitwiseShiftLayerCPUTest::generate_eltwise_input(const ov::element::Type& type,
                                                            const ov::Shape& shape,
                                                            const size_t in_idx) {
    ov::AnyMap any_map = std::get<4>(GetParam());
    const uint32_t max_val = any_map["max_val"].as<uint32_t>();
    if (in_idx == 1) {
        auto shifts_any = any_map["shift"];
        std::vector<int32_t> shifts_vec = shifts_any.as<std::vector<int32_t>>();
        shift_const = std::make_shared<ov::op::v0::Constant>(type, shape, shifts_vec);
        return shift_const->get_tensor_view();
    }
    return ov::test::utils::create_and_fill_tensor_consistently(type, shape, max_val + 1, 0, 1);
}

void BitwiseShiftLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (size_t i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        inputs.insert({funcInput.get_node_shared_ptr(),
                       generate_eltwise_input(funcInput.get_element_type(), targetInputStaticShapes[i], i)});
    }
}

void BitwiseShiftLayerCPUTest::SetUp() {
    EltwiseTestParams basicParamsSet;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    bool enforceSnippets;
    ov::AnyMap val_map;
    std::tie(basicParamsSet, cpuParams, fusingParams, enforceSnippets, val_map) = this->GetParam();
    std::vector<InputShape> shapes;
    ElementType netType;
    utils::InputLayerType secondaryInputType;
    ov::test::utils::OpType opType;
    ov::AnyMap additionalConfig;

    std::tie(shapes,
             eltwiseType,
             secondaryInputType,
             opType,
             netType,
             inType,
             outType,
             targetDevice,
             additionalConfig) = basicParamsSet;
    abs_threshold = 0;

    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    std::tie(postOpMgrPtr, fusedOps) = fusingParams;

    shapes.resize(2);
    switch (opType) {
    case ov::test::utils::OpType::SCALAR: {
        std::vector<ov::Shape> identityShapes(shapes[0].second.size(), {1});
        shapes[1] = {{}, identityShapes};
        break;
    }
    case ov::test::utils::OpType::VECTOR:
        if (shapes[1].second.empty()) {
            shapes[1] = shapes[0];
        }
        break;
    default:
        FAIL() << "Unsupported Secondary operation type";
    }

    init_input_shapes(shapes);
    configuration.insert(additionalConfig.begin(), additionalConfig.end());
    updateSelectedType("ref", netType, configuration);

    if (enforceSnippets) {
        configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::IGNORE_CALLBACK));
    } else {
        configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::DISABLE));
    }

    ov::ParameterVector parameters{std::make_shared<ov::op::v0::Parameter>(netType, inputDynamicShapes.front())};
    std::shared_ptr<ov::Node> secondaryInput;
    switch (secondaryInputType) {
    case utils::InputLayerType::PARAMETER: {
        auto param = std::make_shared<ov::op::v0::Parameter>(netType, inputDynamicShapes.back());
        secondaryInput = param;
        parameters.push_back(param);
        break;
    }
    case utils::InputLayerType::CONSTANT: {
        auto pShape = inputDynamicShapes.back();
        ov::Shape shape;
        if (pShape.is_static()) {
            shape = pShape.get_shape();
        } else {
            ASSERT_TRUE(pShape.rank().is_static());
            shape = std::vector<size_t>(pShape.rank().get_length(), 1);
            for (size_t i = 0; i < pShape.size(); ++i) {
                if (pShape[i].is_static()) {
                    shape[i] = pShape[i].get_length();
                }
            }
        }

        std::ignore = generate_eltwise_input(netType, shape, 1);
        secondaryInput = shift_const;
        break;
    }
    default: {
        FAIL() << "Unsupported InputLayerType";
    }
    }

    auto eltwise = utils::make_eltwise(parameters[0], secondaryInput, eltwiseType);
    function = makeNgraphFunction(netType, parameters, eltwise, "Eltwise");
}

TEST_P(BitwiseShiftLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, std::set<std::string>{"Eltwise", "Subgraph"});
}

}  // namespace test
}  // namespace ov
