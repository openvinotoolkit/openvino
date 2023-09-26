// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conversion.hpp"

#include "gtest/gtest.h"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

std::string ConvertCPULayerTest::getTestCaseName(testing::TestParamInfo<convertLayerTestParamsSet> obj) {
    InputShape inputShape;
    InferenceEngine::Precision inPrc, outPrc;
    CPUSpecificParams cpuParams;
    std::tie(inputShape, inPrc, outPrc, cpuParams) = obj.param;

    std::ostringstream result;

    result << "IS=" << ov::test::utils::partialShape2str({inputShape.first}) << "_";
    result << "TS=";
    for (const auto& shape : inputShape.second) {
        result << ov::test::utils::vec2str(shape) << "_";
    }
    result << "inputPRC=" << inPrc.name() << "_";
    result << "targetPRC=" << outPrc.name() << "_";
    result << CPUTestsBase::getTestCaseName(cpuParams);

    return result.str();
}

bool ConvertCPULayerTest::isInOutPrecisionSupported(InferenceEngine::Precision inPrc, InferenceEngine::Precision outPrc) {
    // WA: I32 precision support disabled in snippets => primitive has to be changed
    // TODO: remove the WA after I32 is supported in snippets (ticket: 99803)
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
    if (inPrc == InferenceEngine::Precision::I32 || outPrc == InferenceEngine::Precision::I32)
        return false;
#endif
    // ACL does not support specific in-out precision pairs
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    if ((inPrc == InferenceEngine::Precision::I8 && outPrc == InferenceEngine::Precision::U8) ||
        (inPrc == InferenceEngine::Precision::U8 && outPrc == InferenceEngine::Precision::I8) ||
        (inPrc == InferenceEngine::Precision::FP32 && (outPrc == InferenceEngine::Precision::U8 ||
                                                       outPrc == InferenceEngine::Precision::I8)))
            return false;
#endif
    return true;
}

void ConvertCPULayerTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;

    InputShape shapes;
    CPUSpecificParams cpuParams;
    std::tie(shapes, inPrc, outPrc, cpuParams) = GetParam();

    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    auto primitive = selectedType;
    if (primitive.empty())
        primitive = getPrimitiveType();
    if (!isInOutPrecisionSupported(inPrc, outPrc))
        primitive = "ref";

    auto exec_type_precision = inPrc != InferenceEngine::Precision::U8
                                    ? inPrc
                                    : InferenceEngine::Precision(InferenceEngine::Precision::I8);
    selectedType = makeSelectedTypeStr(primitive, InferenceEngine::details::convertPrecision(exec_type_precision));

    for (size_t i = 0; i < shapes.second.size(); i++) {
        targetStaticShapes.push_back(std::vector<ngraph::Shape>{shapes.second[i]});
    }

    inputDynamicShapes.push_back(shapes.first);

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);
    auto targetPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(outPrc);
    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, shape));
    }
    auto conversion = ngraph::builder::makeConversion(params.front(), targetPrc, helpers::ConversionTypes::CONVERT);

    function = makeNgraphFunction(ngPrc, params, conversion, "ConversionCPU");
}

void ConvertCPULayerTest::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    if (outPrc != Precision::BOOL) {
        SubgraphBaseTest::generate_inputs(targetInputStaticShapes);
        return;
    }

    // In the scenario where input precision is floating point and output precision is boolean,
    // for CPU plugin, the output precision boolean will be converted to u8 during common transformation,
    // the elements in the output tensor will retain the format of u8 with the range [0, 255].
    // But the output precision in ngraph reference is literal boolean, the elements are either 0 or 1.
    // Here input floating points values are set to be in the range of [-1, 1], so no extra precision
    // converting between actual output and expected output will be needed from the side of single layer tests.
    inputs.clear();
    const auto& funcInputs = function->inputs();

    auto shape = targetInputStaticShapes.front();
    size_t size = shape_size(shape);
    ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(), shape, 2 * size);

    if (inPrc == Precision::FP32) {
        auto* rawBlobDataPtr = static_cast<float*>(tensor.data());
        for (size_t i = 0; i < size; ++i) {
            rawBlobDataPtr[i] = rawBlobDataPtr[i] / size - 1;
        }
    } else if (inPrc == Precision::BF16) {
        auto* rawBlobDataPtr = static_cast<ngraph::bfloat16*>(tensor.data());
        for (size_t i = 0; i < size; ++i) {
            rawBlobDataPtr[i] = rawBlobDataPtr[i] / size - 1;
        }
    } else {
        FAIL() << "Generating inputs with precision" << inPrc << " isn't supported, if output precision is boolean.";
    }

    inputs.insert({funcInputs[0].get_node_shared_ptr(), tensor});
}

TEST_P(ConvertCPULayerTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, std::set<std::string>{"Convert", "Subgraph"});
}

namespace Conversion {

const std::vector<InputShape>& inShapes_4D_static() {
    static const std::vector<InputShape> inShapes_4D_static = {
        {{1, 2, 3, 4}, {{1, 2, 3, 4}}},
        {{1, 1, 1080, 1920}, {{1, 1, 1080, 1920}}},
    };
    return inShapes_4D_static;
}

const std::vector<InputShape>& inShapes_7D_static() {
    static const std::vector<InputShape> inShapes_7D_static = {
        {{1, 2, 3, 4, 5, 6, 7}, {{1, 2, 3, 4, 5, 6, 7}}},
        {{1, 1, 1, 1, 1, 1080, 1920}, {{1, 1, 1, 1, 1, 1080, 1920}}},
    };
    return inShapes_7D_static;
}

const std::vector<InputShape>& inShapes_4D_dynamic() {
    static const std::vector<InputShape> inShapes_4D_dynamic = {
            {
                // dynamic
                {{-1, -1, -1, -1}},
                // target
                {
                    {2, 4, 4, 1},
                    {2, 17, 5, 4},
                    {1, 2, 3, 4}
                }
            },
            {
                // dynamic
                {{{1, 5}, {2, 22}, {2, 9}, {1, 4}}},
                // target
                {
                    {2, 17, 5, 4},
                    {5, 2, 3, 2},
                    {1, 10, 4, 1},
                }
            }
    };
    return inShapes_4D_dynamic;
}

const std::vector<InputShape>& inShapes_7D_dynamic() {
    static const std::vector<InputShape> inShapes_7D_dynamic = {
            {
                // dynamic
                {{-1, -1, -1, -1, -1, -1, -1}},
                // target
                {
                    {2, 4, 4, 4, 3, 3, 1},
                    {2, 17, 5, 4, 3, 2, 1},
                    {1, 2, 3, 4, 5, 6, 7}
                }
            },
            {
                // dynamic
                {{{1, 5}, {2, 22}, {2, 9}, {1, 4}, {1, 4}, {1, 4}, {1, 4}}},
                // target
                {
                    {2, 17, 5, 4, 3, 1, 2},
                    {5, 2, 3, 2, 4, 1, 3},
                    {1, 10, 4, 1, 4, 2, 3},
                }
            }
    };
    return inShapes_7D_dynamic;
}

const std::vector<Precision>& precisions() {
    static const std::vector<Precision> precisions = {
            Precision::U8,
            Precision::I8,
            Precision::I32,
            Precision::FP32,
            Precision::BF16
    };
    return precisions;
}

}  // namespace Conversion
}  // namespace CPULayerTestsDefinitions