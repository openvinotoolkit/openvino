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

static std::string special_value_to_string(const ov::test::SpecialValue& value) {
    if (value == SpecialValue::none) {
        return "none";
    } else if (value == SpecialValue::nan) {
        return "nan";
    } else if (value == SpecialValue::inf) {
        return "inf";
    } else if (value == SpecialValue::overflow) {
        return "overflow";
    }
    return "unknown";
}

template <typename T>
static T set_special_value(T& value, const ov::test::SpecialValue& special_value) {
    if (special_value == ov::test::SpecialValue::nan) {
        value = NAN;
    } else if (special_value == ov::test::SpecialValue::inf) {
        value = INFINITY;
    } else if (special_value == ov::test::SpecialValue::overflow) {
        value = value + std::numeric_limits<ov::float8_e5m2>::max();
    }
    return value;
}

template <typename T>
static void modify_value(ov::Tensor& tensor, const ov::test::SpecialValue& special_value) {
    T* dataPtr = static_cast<T*>(tensor.data());
    for (size_t i = 0; i < tensor.get_size(); i++) {
        set_special_value<T>(dataPtr[i], special_value);
    }
}

std::string ConvertCPULayerTest::getTestCaseName(testing::TestParamInfo<convertLayerTestParamsSet> obj) {
    InputShape inputShape;
    ov::element::Type inPrc, outPrc;
    ov::test::SpecialValue special_value;
    CPUSpecificParams cpuParams;
    std::tie(inputShape, inPrc, outPrc, special_value, cpuParams) = obj.param;

    std::ostringstream result;

    result << "IS=" << ov::test::utils::partialShape2str({inputShape.first}) << "_";
    result << "TS=";
    for (const auto& shape : inputShape.second) {
        result << ov::test::utils::vec2str(shape) << "_";
    }
    result << "inputPRC=" << inPrc.to_string() << "_";
    result << "specialValue=" << special_value_to_string(special_value) << "_";
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
        (inPrc == ov::element::f32 && (outPrc == ov::element::u8 || outPrc == ov::element::i8)) ||
        (inPrc == ov::element::f8e4m3 || inPrc == ov::element::f8e5m2 ||
        outPrc == ov::element::f8e4m3 || outPrc == ov::element::f8e5m2))
        return false;
#endif
    return true;
}

void ConvertCPULayerTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;

    InputShape shapes;
    CPUSpecificParams cpuParams;
    std::tie(shapes, inPrc, outPrc, special_value, cpuParams) = GetParam();

    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    auto primitive = selectedType;
    if (primitive.empty())
        primitive = getPrimitiveType();
#if defined(OPENVINO_ARCH_ARM64)
    if (inPrc == ov::element::u4 || inPrc == ov::element::i4 ||
        inPrc == ov::element::f8e4m3 || inPrc == ov::element::f8e5m2 ||
        outPrc == ov::element::f8e4m3 || outPrc == ov::element::f8e5m2 ||
        outPrc == ov::element::nf4) {
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

    if (outPrc == ov::element::f16) {
        configuration.insert(ov::hint::inference_precision(ov::element::f16));
    } else if (outPrc == ov::element::bf16) {
        configuration.insert(ov::hint::inference_precision(ov::element::bf16));
    }

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(inPrc, shape));
    }
    auto conversion = std::make_shared<ov::op::v0::Convert>(params.front(), outPrc);

    function = makeNgraphFunction(inPrc, params, conversion, "ConversionCPU");

    // issue 161636
    if (special_value == ov::test::SpecialValue::none && outPrc == ov::element::f8e4m3) {
        abs_threshold = 0.0078125f;
        rel_threshold = 1e-2f;
    }
}

void ConvertCPULayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    if (outPrc != ov::element::nf4 && special_value == ov::test::SpecialValue::none) {
        SubgraphBaseTest::generate_inputs(targetInputStaticShapes);
        return;
    }

    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (size_t i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        ov::Tensor tensor;
        if (outPrc == ov::element::nf4) {
            tensor = ov::test::utils::create_and_fill_tensor_real_distribution(funcInput.get_element_type(),
                                                                               targetInputStaticShapes[i],
                                                                               -1.f,
                                                                               1.f,
                                                                               1);
        } else {
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
        }

        if (inPrc == ov::element::f32) {
            modify_value<float>(tensor, special_value);
        } else if (inPrc == ov::element::f16) {
            modify_value<ov::float16>(tensor, special_value);
        } else if (inPrc == ov::element::bf16) {
            modify_value<ov::bfloat16>(tensor, special_value);
        } else if (inPrc == ov::element::f8e4m3) {
            modify_value<ov::float8_e4m3>(tensor, special_value);
        } else if (inPrc == ov::element::f8e5m2) {
            modify_value<ov::float8_e5m2>(tensor, special_value);
        }

        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
}

void ConvertCPULayerTest::validate_out_prc() const {
    if (outPrc == ov::element::boolean)
        FAIL() << "ConvertCPULayerTest supports only non boolean output prc";
}

void ConvertCPULayerTest::validate() {
    if (outPrc == ov::element::nf4) {
        // Use custom bit-exact validation, because common tests infra doesn't support 4bits tensors comparision
        auto actualOutputs = get_plugin_outputs();
        auto expectedOutputs = calculate_refs();
        ASSERT_EQ(expectedOutputs.size(), actualOutputs.size());
        ASSERT_EQ(expectedOutputs.size(), 1);
        ASSERT_EQ(expectedOutputs[0].get_shape(), actualOutputs[0].get_shape());
        ASSERT_EQ(expectedOutputs[0].get_element_type(), ov::element::nf4);
        ASSERT_EQ(expectedOutputs[0].get_element_type(), actualOutputs[0].get_element_type());

        auto expected_data = reinterpret_cast<const uint8_t*>(expectedOutputs[0].data());
        auto actual_data = reinterpret_cast<const uint8_t*>(actualOutputs[0].data());
        size_t byte_count = shape_size(expectedOutputs[0].get_shape()) / 2;
        bool has_tile = shape_size(expectedOutputs[0].get_shape()) % 2 != 0;
        for (size_t i = 0; i < byte_count; ++i) {
            uint8_t expected_value = expected_data[i];
            uint8_t actual_value = actual_data[i];
            ASSERT_EQ(expected_value, actual_value);
        }

        // Convert operation doc doesn't specify behavior for odd amount of elements: should upper 4 bits of last byte be filled with zeros or not.
        // CPU Plugin fills these bits with zeros as it better fits optimized kernels which get NF4 inputs.
        // In general it is considered as UB, so skip the check for last 4 bits.
        if (has_tile) {
            ASSERT_EQ(expected_data[byte_count] & 0x0F, actual_data[byte_count] & 0x0F);
        }

        return;
    }

    SubgraphBaseTest::validate();
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
                    {1, 2, 3, 4},
                    // odd number of elements
                    {1, 3, 3, 3}
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