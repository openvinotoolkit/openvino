// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conversion.hpp"

#include "gtest/gtest.h"
#include "internal_properties.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "shared_test_classes/base/utils/compare_results.hpp"


using namespace CPUTestUtils;

namespace ov {
namespace test {

std::string ConvertCPULayerTest::getTestCaseName(testing::TestParamInfo<convertLayerTestParamsSet> obj) {
    InputShape inputShape;
    ov::element::Type inPrc, outPrc;
    CPUSpecificParams cpuParams;
    std::tie(inputShape, inPrc, outPrc, cpuParams) = obj.param;

    std::ostringstream result;

    result << "IS=" << ov::test::utils::partialShape2str({inputShape.first}) << "_";
    result << "TS=";
    for (const auto& shape : inputShape.second) {
        result << ov::test::utils::vec2str(shape) << "_";
    }
    result << "inputPRC=" << inPrc.to_string() << "_";
    result << "targetPRC=" << outPrc.to_string() << "_";
    result << CPUTestsBase::getTestCaseName(cpuParams);

    return result.str();
}

bool ConvertCPULayerTest::isInOutPrecisionSupported(ov::element::Type inPrc, ov::element::Type outPrc) {
    // WA: I32 precision support disabled in snippets => primitive has to be changed
    // TODO: remove the WA after I32 is supported in snippets (ticket: 99803)
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
    if (inPrc == ov::element::i32 || outPrc == ov::element::i32)
        return false;
#endif
        // ACL does not support specific in-out precision pairs
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    if ((inPrc == ov::element::i8 && outPrc == ov::element::u8) ||
        (inPrc == ov::element::u8 && outPrc == ov::element::i8) ||
        (inPrc == ov::element::f32 && (outPrc == ov::element::u8 || outPrc == ov::element::i8)))
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
#if defined(OPENVINO_ARCH_ARM64)
    if (inPrc == ov::element::u4 || inPrc == ov::element::i4) {
        primitive = "ref";
    } else if (shapes.first.is_static() &&
        inPrc != ov::element::bf16 && outPrc != ov::element::bf16 &&
        inPrc != ov::element::i32 && outPrc != ov::element::i32) { // Apply "jit" for the snippets cases
        primitive = "jit";
        if (shapes.first.rank().get_length() > 6) {
            configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::IGNORE_CALLBACK));
        }
    } else {
        primitive = "acl";
    }
#endif
    if (primitive != "jit" && !isInOutPrecisionSupported(inPrc, outPrc))
        primitive = "ref";

    validate_out_prc();

    auto exec_type_precision = inPrc != ov::element::u8 ? inPrc : ov::element::Type(ov::element::i8);
    selectedType = makeSelectedTypeStr(primitive, exec_type_precision);

    for (size_t i = 0; i < shapes.second.size(); i++) {
        targetStaticShapes.push_back(std::vector<ov::Shape>{shapes.second[i]});
    }

    inputDynamicShapes.push_back(shapes.first);

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(inPrc, shape));
    }
    auto conversion = std::make_shared<ov::op::v0::Convert>(params.front(), outPrc);

    function = makeNgraphFunction(inPrc, params, conversion, "ConversionCPU");
}

void ConvertCPULayerTest::validate_out_prc() const {
    if (outPrc == ov::element::boolean)
        FAIL() << "ConvertCPULayerTest supports only non boolean output prc";
}

void ConvertToBooleanCPULayerTest::validate_out_prc() const {
    if (outPrc != ov::element::boolean)
        FAIL() << "ConvertToBooleanCPULayerTest supports only boolean output prc";
}

void ConvertToBooleanCPULayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();

    auto shape = targetInputStaticShapes.front();
    auto size = shape_size(shape);

    ov::Tensor tensor = ov::Tensor(funcInputs[0].get_element_type(), shape);
    const auto first_part_size = size / 2;
    const auto second_part_size = size - first_part_size;

    // 1). Validate the nearest to zero values (Abs + Ceil)
    {
        double start_from = -2;
        uint32_t range = 4;
        int32_t resolution = size;
        if (inPrc == ov::element::f32) {
            auto* rawBlobDataPtr = static_cast<float*>(tensor.data());
            ov::test::utils::fill_data_random(rawBlobDataPtr, first_part_size, range, start_from, resolution);
        } else if (inPrc == ov::element::bf16) {
            auto* rawBlobDataPtr = static_cast<ov::bfloat16*>(tensor.data());
            ov::test::utils::fill_data_random(rawBlobDataPtr, first_part_size, range, start_from, resolution);
        } else {
            FAIL() << "Generating inputs with precision " << inPrc.to_string() << " isn't supported, if output precision is boolean.";
        }
    }

    // 2). Validate the values that are more than UINT8_MAX in absolute (Abs + Min)
    {
        ov::test::utils::InputGenerateData in_data_neg;
        double neg_start_from = -1.5 * std::numeric_limits<uint8_t>::max();
        double pos_start_from = 0.5 * std::numeric_limits<uint8_t>::max();
        uint32_t range = 256;
        auto neg_size = second_part_size / 2;
        auto pos_size = second_part_size - neg_size;
        int32_t resolution = 1;

        if (inPrc == ov::element::f32) {
            auto* rawBlobDataPtr = static_cast<float*>(tensor.data());
            ov::test::utils::fill_data_random(rawBlobDataPtr + first_part_size, neg_size, range, neg_start_from, resolution);
            ov::test::utils::fill_data_random(rawBlobDataPtr + first_part_size + neg_size, pos_size, range, pos_start_from, resolution);
        } else if (inPrc == ov::element::bf16) {
            auto* rawBlobDataPtr = static_cast<ov::bfloat16*>(tensor.data());
            ov::test::utils::fill_data_random(rawBlobDataPtr + first_part_size, neg_size, range, neg_start_from, resolution);
            ov::test::utils::fill_data_random(rawBlobDataPtr + first_part_size + neg_size, pos_size, range, pos_start_from, resolution);
        } else {
            FAIL() << "Generating inputs with precision " << inPrc.to_string() << " isn't supported, if output precision is boolean.";
        }
    }

    inputs.insert({funcInputs[0].get_node_shared_ptr(), tensor});
}

TEST_P(ConvertCPULayerTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, std::set<std::string>{"Convert", "Subgraph"});
}

TEST_P(ConvertToBooleanCPULayerTest, CompareWithRefs) {
    run();
    // CPU Plugin decomposes Convert[...->BOOL] into Convert[...->supported] + Abs + Min + Seiling + Convert[supported->u8].
    // To align output precision of model, Plugin insertes Convert[U8->Boolean] on output.
    // To avoid conflicts of mapping node types and prim types in CheckPluginRelatedResults, we set empty set of node types
    CheckPluginRelatedResults(compiledModel, std::set<std::string>{});
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

const std::vector<ov::element::Type>& precisions() {
    static const std::vector<ov::element::Type> precisions = {
            ov::element::u8,
            ov::element::i8,
            ov::element::i32,
            ov::element::f32,
            ov::element::bf16
    };
    return precisions;
}

}  // namespace Conversion
}  // namespace test
}  // namespace ov