// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/src/classes/eltwise_chain.hpp"

#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
using namespace ov::test::utils;
using namespace ov::test::eltwise_chain;

namespace {

std::vector<std::vector<EltwiseChainOp>> eltwiseOpsConvertInt8 = {
        { EltwiseTypes::MULTIPLY },
        { EltwiseTypes::ADD },
        { EltwiseTypes::DIVIDE }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_EltwiseChain_MergeConvert_int8,
    EltwiseChainTest,
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesConvert())),
                       ::testing::Values(InputLayerType::CONSTANT),
                       ::testing::ValuesIn(inputPrecisionsConvert()),
                       ::testing::ValuesIn(eltwiseOpsConvertInt8),
                       ::testing::Values(PostNode::EMPTY),
                       ::testing::ValuesIn({ov::element::i8, ov::element::u8}),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(std::nullopt)),
    EltwiseChainTest::getTestCaseName);

// Reuse the chain builder for exact i32 arithmetic, fused comparisons and conversions.
// Separate narrow input windows keep every intermediate within the i32 range.
std::vector<InputShape> precisionShapes(size_t shapeCase, size_t inputs) {
    std::vector<InputShape> shapes;
    switch (shapeCase) {
    case 0:
        shapes = {{{}, {{1, 1}}}, {{}, {{1, 1}}}};
        break;
    case 1:
        shapes = {{{}, {{1, 4}}}, {{}, {{1, 4}}}};
        break;
    case 2:
    case 3:
    case 4: {
        const size_t length = shapeCase == 2 ? 29 : shapeCase == 3 ? 31 : 37;
        shapes = {{{}, {{1, length}}}, {{}, {{1, length}}}};
        break;
    }
    case 5:
        shapes = {{{}, {{2, 29}}}, {{}, {{1, 1}}}};
        break;
    case 6:
        shapes = {{{-1, -1}, {{1, 29}, {3, 29}, {5, 1}, {2, 29}}}, {{1, -1}, {{1, 29}, {1, 29}, {1, 1}, {1, 29}}}};
        break;
    case 7:
        // Rank seven exercises the generic fallback with default Snippets settings.
        shapes = {{{}, {{1, 1, 1, 1, 1, 1, 29}}}, {{}, {{1, 1, 1, 1, 1, 1, 29}}}};
        break;
    default:
        OPENVINO_THROW("Unexpected precision shape case");
    }
    shapes.resize(inputs, shapes.back());
    return shapes;
}

EltwiseChainTuple precisionCase(const std::vector<EltwiseChainOp>& ops,
                                const std::vector<ElementType>& types,
                                size_t shapeCase,
                                bool constant,
                                EltwiseChainConfig config,
                                PostNode postNode = PostNode::EMPTY,
                                ov::element::Type conversion = ov::element::dynamic) {
    return {precisionShapes(shapeCase, types.size()),
            constant ? InputLayerType::CONSTANT : InputLayerType::PARAMETER,
            types,
            ops,
            postNode,
            conversion,
            DEVICE_CPU,
            std::move(config)};
}

std::vector<EltwiseChainTuple> integerPrecisionCases() {
    const std::vector<std::pair<std::string, std::vector<EltwiseChainOp>>> operations = {
        {"Add", {EltwiseTypes::ADD}},
        {"Subtract", {EltwiseTypes::SUBTRACT}},
        {"Multiply", {EltwiseTypes::MULTIPLY}},
        {"SquaredDifference", {EltwiseTypes::SQUARED_DIFF}},
        {"Minimum", {MinMaxOpType::MINIMUM}},
        {"Maximum", {MinMaxOpType::MAXIMUM}},
        {"Abs", {ActivationTypes::Abs}},
        {"Negative", {ActivationTypes::Negative}},
        {"Floor", {ActivationTypes::Floor}},
        {"MulAdd", {EltwiseTypes::MULTIPLY, EltwiseTypes::ADD}},
        {"MulAddAfterAdd", {EltwiseTypes::ADD, EltwiseTypes::MULTIPLY, EltwiseTypes::ADD}},
        {"ArithmeticChain", {EltwiseTypes::ADD, EltwiseTypes::SUBTRACT}},
        {"DivideAfterAdd", {EltwiseTypes::ADD, EltwiseTypes::DIVIDE}},
    };
    std::vector<EltwiseChainTuple> cases;
    for (const auto& [name, ops] : operations) {
        const bool unary = std::holds_alternative<ActivationTypes>(ops.front());
        for (const auto type : {ElementType::i32, ElementType::f32}) {
            const std::vector<double> starts =
                type == ElementType::i32 ? std::vector<double>{(1 << 24) + 1, -(1 << 24) - 9} : std::vector<double>{-9};
            for (size_t window = 0; window < starts.size(); ++window) {
                for (size_t shapeCase = 0; shapeCase <= 6; ++shapeCase) {
                    for (const bool constant : {false, true}) {
                        if (unary && constant) {
                            continue;
                        }
                        for (const bool snippets : {false, true}) {
                            if (name == "DivideAfterAdd" && snippets) {
                                continue;
                            }
                            EltwiseChainConfig config;
                            config.name =
                                name + "_Window" + std::to_string(window) + "_Snippets" + std::to_string(snippets);
                            config.snippetsMode = snippets ? ov::intel_cpu::SnippetsMode::IGNORE_CALLBACK
                                                           : ov::intel_cpu::SnippetsMode::DISABLE;
                            config.selectedType = type == ElementType::i32 ? "jit_i32" : "jit_f32";
                            config.inputData.assign(unary ? 1 : ops.size() + 1, InputGenerateData(1, 2));
                            config.inputData[0] = InputGenerateData(starts[window], 8);
                            if (name == "SquaredDifference" || name == "Minimum" || name == "Maximum") {
                                config.inputData[1] = InputGenerateData(starts[window], 8, 1, 2);
                            }
                            config.expectedNodes = {{snippets ? "Subgraph" : "Eltwise",
                                                     name == "ArithmeticChain" && !constant && !snippets ? 2 : 1}};
                            if (name == "DivideAfterAdd" && type == ElementType::f32 && !constant) {
                                config.expectedNodes = {{"Eltwise", 2}};
                            }
                            if (name == "DivideAfterAdd" && type == ElementType::i32) {
                                config.selectedType = "(jit|ref)_i32";
                                // Constant divisors can be folded or decomposed.
                                config.expectedNodes.clear();
                                if (!constant) {
                                    config.expectedImplementations = {"jit_i32", "ref_i32"};
                                    config.expectedNodes = {{"Eltwise", 3}};
                                }
                            }
                            cases.push_back(precisionCase(ops,
                                                          std::vector<ElementType>(config.inputData.size(), type),
                                                          shapeCase,
                                                          constant,
                                                          config));
                        }
                    }
                }
            }
        }
    }
    // Near-limit arithmetic must stay representable, including all fused intermediates.
    for (const auto& [name, ops] : operations) {
        if (name == "SquaredDifference" || name == "DivideAfterAdd") {
            continue;
        }
        const bool unary = std::holds_alternative<ActivationTypes>(ops.front());
        for (const bool negative : {false, true}) {
            for (const bool snippets : {false, true}) {
                EltwiseChainConfig config;
                config.name = name + "_Limit" + std::to_string(negative) + "_Snippets" + std::to_string(snippets);
                config.snippetsMode =
                    snippets ? ov::intel_cpu::SnippetsMode::IGNORE_CALLBACK : ov::intel_cpu::SnippetsMode::DISABLE;
                const double bound =
                    negative ? std::numeric_limits<int32_t>::min() + 8.0 : std::numeric_limits<int32_t>::max() - 16.0;
                config.inputData.assign(unary ? 1 : ops.size() + 1, InputGenerateData(1, 0));
                const bool multiply = name == "Multiply" || name == "MulAdd" || name == "MulAddAfterAdd";
                config.inputData[0] = InputGenerateData(multiply ? bound / 2 : bound, 4);
                if (multiply) {
                    config.inputData[name == "MulAddAfterAdd" ? 2 : 1] = InputGenerateData(2, 0);
                }
                if (name == "Minimum" || name == "Maximum") {
                    config.inputData[1] = InputGenerateData(bound, 4, 1, 2);
                }
                config.expectedNodes = {
                    {snippets ? "Subgraph" : "Eltwise", name == "ArithmeticChain" && !snippets ? 2 : 1}};
                cases.push_back(precisionCase(ops,
                                              std::vector<ElementType>(config.inputData.size(), ElementType::i32),
                                              2,
                                              false,
                                              config));
            }
        }
    }
    return cases;
}

std::vector<EltwiseChainTuple> comparisonPrecisionCases() {
    std::vector<EltwiseChainTuple> cases;
    for (const auto comparison : {ComparisonTypes::EQUAL,
                                  ComparisonTypes::NOT_EQUAL,
                                  ComparisonTypes::GREATER,
                                  ComparisonTypes::GREATER_EQUAL,
                                  ComparisonTypes::LESS,
                                  ComparisonTypes::LESS_EQUAL}) {
        for (const bool negate : {false, true}) {
            for (const size_t shapeCase : {0, 2, 6}) {
                for (const bool constant : {false, true}) {
                    for (const bool snippets : {false, true}) {
                        const std::vector<double> starts = {(1 << 24) + 1,
                                                            -(1 << 24) - 9,
                                                            std::numeric_limits<int32_t>::max() - 16.0,
                                                            std::numeric_limits<int32_t>::min() + 8.0};
                        for (size_t window = 0; window < starts.size(); ++window) {
                            const double start = starts[window];
                            EltwiseChainConfig config;
                            config.selectedType = negate ? "jit_(i32|I8)" : "jit_i32";
                            config.name =
                                "Comparison_Window" + std::to_string(window) + "_Snippets" + std::to_string(snippets);
                            config.inputData = {InputGenerateData(start, 8),
                                                InputGenerateData(2, 0),
                                                InputGenerateData(1, 0),
                                                InputGenerateData(start, 8, 1, 2)};
                            config.snippetsMode = snippets ? ov::intel_cpu::SnippetsMode::IGNORE_CALLBACK
                                                           : ov::intel_cpu::SnippetsMode::DISABLE;
                            // Parameter operands keep Add separate; LogicalNot consumes the boolean output separately.
                            config.expectedNodes = {{snippets ? "Subgraph" : "Eltwise",
                                                     snippets ? 1 : (constant ? 1 : 2) + size_t(negate)}};
                            cases.push_back(precisionCase({EltwiseTypes::ADD, EltwiseTypes::SUBTRACT, comparison},
                                                          std::vector<ElementType>(4, ElementType::i32),
                                                          shapeCase,
                                                          constant,
                                                          config,
                                                          negate ? PostNode::LOGICAL_NOT : PostNode::EMPTY));
                        }
                    }
                }
            }
        }
    }
    return cases;
}

std::vector<EltwiseChainTuple> conversionPrecisionCases() {
    std::vector<EltwiseChainTuple> cases;
    for (const auto sourceType : {ElementType::u8, ElementType::i8, ElementType::f16, ElementType::f32}) {
        for (const bool floor : {false, true}) {
            if (sourceType == ElementType::f32 && !floor) {
                continue;
            }
            if (floor && sourceType != ElementType::f16 && sourceType != ElementType::f32) {
                continue;
            }
            for (const size_t shapeCase : {0, 1, 2, 5, 6, 7}) {
                for (size_t port = 0; port < (floor ? 1 : 2); ++port) {
                    for (const bool disableSnippets : {false, true}) {
                        EltwiseChainConfig config;
                        config.name = std::string(floor ? "Floor" : "Multiply") + "_ConvertPort" +
                                      std::to_string(port) + "_DisableSnippets" + std::to_string(disableSnippets);
                        config.snippetsMode = disableSnippets ? ov::intel_cpu::SnippetsMode::DISABLE
                                                              : ov::intel_cpu::SnippetsMode::ENABLE;
                        std::vector<ElementType> types(floor ? 1 : 2, ElementType::i32);
                        types[port] = sourceType;
                        config.inputData.assign(types.size(), InputGenerateData((1 << 23) + 1, 0));
                        config.inputData[port] = sourceType == ElementType::u8   ? InputGenerateData(0, 256)
                                                 : sourceType == ElementType::i8 ? InputGenerateData(-128, 256)
                                                                                 : InputGenerateData(-127.75, 256, 4);
                        config.selectedType =
                            port == 0 && (sourceType == ElementType::i8 || sourceType == ElementType::u8)
                                ? "jit_(I8|i32)"
                                : "jit_i32";
                        // Default tokenization can choose either integer implementation.
                        if (sourceType == ElementType::f16 || sourceType == ElementType::f32) {
                            // Fractional values must be converted before integer execution.
                            config.expectedNodes = {{"Convert", 1}};
                        } else if (disableSnippets || shapeCase == 7) {
                            config.expectedNodes = {{"Convert", 0}, {"Eltwise", 1}};
                        }
                        const std::vector<EltwiseChainOp> ops =
                            floor ? std::vector<EltwiseChainOp>{ActivationTypes::Floor}
                                  : std::vector<EltwiseChainOp>{EltwiseTypes::MULTIPLY};
                        auto testCase =
                            precisionCase(ops, types, shapeCase, false, config, PostNode::EMPTY, ov::element::i32);
                        if (!floor && port == 0 && shapeCase == 5) {
                            std::swap(std::get<0>(testCase)[0], std::get<0>(testCase)[1]);
                        }
                        cases.push_back(std::move(testCase));
                    }
                }
            }
        }
    }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseChain_I32Precision,
                         EltwiseChainTest,
                         ::testing::ValuesIn(integerPrecisionCases()),
                         EltwiseChainTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_EltwiseChain_I32Comparison,
                         EltwiseChainTest,
                         ::testing::ValuesIn(comparisonPrecisionCases()),
                         EltwiseChainTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_EltwiseChain_I32Conversion,
                         EltwiseChainTest,
                         ::testing::ValuesIn(conversionPrecisionCases()),
                         EltwiseChainTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
