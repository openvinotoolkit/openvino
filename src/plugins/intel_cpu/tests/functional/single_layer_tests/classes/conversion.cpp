// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conversion.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

std::string ConvertCPULayerTest::getTestCaseName(testing::TestParamInfo<convertLayerTestParamsSet> obj) {
    InputShape inputShape;
    ElementType inPrc, outPrc;
    CPUSpecificParams cpuParams;
    ov::AnyMap config;
    std::tie(inputShape, inPrc, outPrc, config, cpuParams) = obj.param;

    std::ostringstream result;

    result << "IS=" << CommonTestUtils::partialShape2str({inputShape.first}) << "_";
    result << "TS=";
    for (const auto& shape : inputShape.second) {
        result << CommonTestUtils::vec2str(shape) << "_";
    }
    result << "inputPRC=" << inPrc << "_";
    result << "targetPRC=" << outPrc << "_";
    result << CPUTestsBase::getTestCaseName(cpuParams);

    if (!config.empty()) {
        result << "_PluginConf";
        for (const auto& configItem : config) {
            result << "_" << configItem.first << "=";
            configItem.second.print(result);
        }
    }

    return result.str();
}

bool ConvertCPULayerTest::isInOutPrecisionSupported(ElementType inPrc, ElementType outPrc) {
    // WA: I32 precision support disabled in snippets => primitive has to be changed
    // TODO: remove the WA after I32 is supported in snippets (ticket: 99803)
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
    if (inPrc == ElementType::i32 || outPrc == ElementType::i32)
        return false;
#endif
    // ACL does not support specific in-out precision pairs
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    if ((inPrc == ElementType::i8 && outPrc == ElementType::u8) ||
        (inPrc == ElementType::u8 && outPrc == ElementType::i8) ||
        (inPrc == ElementType::f32 && (outPrc == ElementType::u8 ||
                                       outPrc == ElementType::i8)))
            return false;
#endif
    return true;
}

void ConvertCPULayerTest::SetUp() {
    targetDevice = CommonTestUtils::DEVICE_CPU;

    InputShape shapes;
    CPUSpecificParams cpuParams;
    std::tie(shapes, inPrc, outPrc, configuration, cpuParams) = GetParam();

    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    auto primitive = selectedType;
    if (primitive.empty())
        primitive = getPrimitiveType();
    // WA: I32 precision support disabled in snippets => primitive has to be changed
    // TODO: remove the WA after I32 is supported in snippets (ticket: 99803)
    if (inPrc == ElementType::i32 || inPrc == ElementType::i64 || outPrc == ElementType::i32 || outPrc == ElementType::i64)
        primitive = "unknown";

    if (inPrc == ElementType::i64 || inPrc == ElementType::u64) {
        auto i64Flag = configuration.find(PluginConfigInternalParams::KEY_CPU_NATIVE_I64);
        if (i64Flag == configuration.end() || i64Flag->second == PluginConfigParams::NO) {
            selectedType = makeSelectedTypeStr(primitive, ElementType::i32);
        } else {
            selectedType = makeSelectedTypeStr(primitive, ElementType::i64);
        }
    } else if (inPrc == ElementType::u8) {
        selectedType = makeSelectedTypeStr(primitive, ElementType::i8);
    } else {
        selectedType = makeSelectedTypeStr(primitive, inPrc);
    }

    for (size_t i = 0; i < shapes.second.size(); i++) {
        targetStaticShapes.push_back(std::vector<ov::Shape>{shapes.second[i]});
    }

    inputDynamicShapes.push_back(shapes.first);

    ov::ParameterVector params = ngraph::builder::makeDynamicParams(inPrc, inputDynamicShapes);
    auto conversion = ngraph::builder::makeConversion(params.front(), outPrc, ngraph::helpers::ConversionTypes::CONVERT);

    function = makeNgraphFunction(inPrc, params, conversion, "ConversionCPU");
}

void ConvertCPULayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    if (outPrc != ElementType::boolean) {
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

    if (inPrc == ElementType::f32) {
        auto* rawBlobDataPtr = static_cast<float*>(tensor.data());
        for (size_t i = 0; i < size; ++i) {
            rawBlobDataPtr[i] = rawBlobDataPtr[i] / size - 1;
        }
    } else if (inPrc == ElementType::bf16) {
        auto* rawBlobDataPtr = static_cast<ov::bfloat16*>(tensor.data());
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

const std::vector<ElementType>& precisions() {
    static const std::vector<ElementType> precisions = {
        ElementType::u8,
        ElementType::i8,
        ElementType::i32,
        ElementType::f32,
        ElementType::bf16
    };
    return precisions;
}

}  // namespace Conversion
}  // namespace CPULayerTestsDefinitions