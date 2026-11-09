// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "internal_properties.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/subtract.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

#if defined(OPENVINO_ARCH_ARM64)
namespace ov::test {
namespace {

using EltwiseI32PrecisionParams = std::tuple<std::string, size_t, bool, bool, ov::element::Type>;

class EltwiseI32PrecisionTest : public testing::WithParamInterface<EltwiseI32PrecisionParams>,
                                public SubgraphBaseTest,
                                public CPUTestUtils::CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<EltwiseI32PrecisionParams>& info) {
        const auto& [operation, shape, constant, snippets, type] = info.param;
        return operation + "_Shape" + std::to_string(shape) + "_Constant" + std::to_string(constant) + "_Snippets" +
               std::to_string(snippets) + "_" + type.to_string();
    }

protected:
    ov::Tensor make_input(const ov::Shape& shape, size_t port) const {
        const auto& [operation, shapeCase, constant, snippets, type] = GetParam();
        constexpr int32_t limit = 1 << 24;
        constexpr int32_t min = std::numeric_limits<int32_t>::min();
        constexpr int32_t max = std::numeric_limits<int32_t>::max();
        constexpr std::array<std::array<int32_t, 3>, 9> arithmetic = {{{limit + 1, 1, 1},
                                                                       {-limit - 1, -1, -1},
                                                                       {max / 2 - 4, 2, 1},
                                                                       {min / 2 + 4, 2, -1},
                                                                       {0, 1, 1},
                                                                       {1, -1, 0},
                                                                       {-1, 1, 0},
                                                                       {limit + 3, 2, 1},
                                                                       {-limit - 3, 2, -1}}};
        constexpr std::array<std::array<int32_t, 2>, 9> extrema = {{{limit + 1, limit},
                                                                    {-limit - 1, -limit},
                                                                    {max, max - 1},
                                                                    {min + 1, min + 2},
                                                                    {0, 0},
                                                                    {1, -1},
                                                                    {-1, 1},
                                                                    {limit, limit + 1},
                                                                    {-limit, -limit - 1}}};
        ov::Tensor tensor(type, shape);
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            int32_t value = arithmetic[i % arithmetic.size()][port];
            if (operation == "Add" || operation == "Subtract" || operation == "ArithmeticChain") {
                if (i % arithmetic.size() == 2) {
                    value = port == 0 ? max - 2 : 1;
                } else if (i % arithmetic.size() == 3) {
                    value = port == 0 ? min + 2 : -1;
                }
            }
            if (operation == "SquaredDifference") {
                // Both operands stay in the same window, including when the second input broadcasts.
                value = limit + static_cast<int32_t>((i + port) % 9);
            } else if (operation == "Minimum" || operation == "Maximum" || operation == "Abs" ||
                       operation == "Negative" || operation == "Floor") {
                value = extrema[i % extrema.size()][port];
            }
            if (type == ov::element::i32) {
                tensor.data<int32_t>()[i] = value;
            } else {
                // Exact small integer operands also cover the unchanged floating-point paths.
                tensor.data<float>()[i] = static_cast<float>(value % 97);
            }
        }
        return tensor;
    }

    void generate_inputs(const std::vector<ov::Shape>& shapes) override {
        inputs.clear();
        const auto& parameters = function->get_parameters();
        for (size_t i = 0; i < parameters.size(); ++i) {
            inputs.emplace(parameters[i], make_input(shapes[i], i));
        }
    }

    void SetUp() override {
        const auto& [operation, shapeCase, constant, snippets, type] = GetParam();
        const bool unary = operation == "Abs" || operation == "Negative" || operation == "Floor";
        const bool mulAdd = operation == "MulAdd" || operation == "MulAddAfterAdd";
        std::vector<InputShape> shapes;
        switch (shapeCase) {
        case 0:
            shapes = {{{}, {{1, 1}}}, {{}, {{1, 1}}}};
            break;
        case 1:
        case 4:
        case 5: {
            const size_t length = shapeCase == 1 ? 29 : shapeCase == 4 ? 31 : 37;
            shapes = {{{}, {{1, length}}}, {{}, {{1, length}}}};
            break;
        }
        case 2:
            shapes = {{{}, {{2, 29}}}, {{}, {{1, 1}}}};
            break;
        case 3:
            shapes = {{{-1, -1}, {{1, 29}, {3, 29}, {5, 1}, {2, 29}}}, {{1, -1}, {{1, 29}, {1, 29}, {1, 1}, {1, 29}}}};
            break;
        default:
            OPENVINO_THROW("Unexpected shape case");
        }
        if (unary) {
            shapes.resize(1);
        } else if (mulAdd) {
            shapes.push_back(shapes[1]);
        }
        init_input_shapes(shapes);
        targetDevice = ov::test::utils::DEVICE_CPU;
        abs_threshold = 0;
        rel_threshold = 0;
        selectedType =
            (operation == "DivideAfterAdd" && type == ov::element::i32 ? "(jit|ref)_" : "jit_") + type.to_string();
        configuration.insert(ov::intel_cpu::snippets_mode(snippets ? ov::intel_cpu::SnippetsMode::IGNORE_CALLBACK
                                                                   : ov::intel_cpu::SnippetsMode::DISABLE));
        ov::ParameterVector parameters;
        std::vector<std::shared_ptr<ov::Node>> operands;
        for (size_t i = 0; i < shapes.size(); ++i) {
            if (i > 0 && constant) {
                operands.push_back(std::make_shared<ov::op::v0::Constant>(make_input(targetStaticShapes[0][i], i)));
            } else {
                auto parameter = std::make_shared<ov::op::v0::Parameter>(type, inputDynamicShapes[i]);
                parameters.push_back(parameter);
                operands.push_back(parameter);
            }
        }
        std::shared_ptr<ov::Node> result;
        if (operation == "Add" || operation == "ArithmeticChain") {
            result = std::make_shared<ov::op::v1::Add>(operands[0], operands[1]);
            if (operation == "ArithmeticChain") {
                result = std::make_shared<ov::op::v1::Subtract>(result, operands[1]);
            }
        } else if (operation == "Subtract") {
            result = std::make_shared<ov::op::v1::Subtract>(operands[0], operands[1]);
        } else if (operation == "Multiply") {
            result = std::make_shared<ov::op::v1::Multiply>(operands[0], operands[1]);
        } else if (operation == "SquaredDifference") {
            result = std::make_shared<ov::op::v0::SquaredDifference>(operands[0], operands[1]);
        } else if (operation == "Minimum") {
            result = std::make_shared<ov::op::v1::Minimum>(operands[0], operands[1]);
        } else if (operation == "Maximum") {
            result = std::make_shared<ov::op::v1::Maximum>(operands[0], operands[1]);
        } else if (operation == "Abs") {
            result = std::make_shared<ov::op::v0::Abs>(operands[0]);
        } else if (operation == "Negative") {
            result = std::make_shared<ov::op::v0::Negative>(operands[0]);
        } else if (operation == "Floor") {
            result = std::make_shared<ov::op::v0::Floor>(operands[0]);
        } else if (operation == "DivideAfterAdd") {
            auto value =
                std::make_shared<ov::op::v1::Add>(operands[0], ov::op::v0::Constant::create(type, ov::Shape{}, {2}));
            result = std::make_shared<ov::op::v1::Divide>(value, operands[1]);
        } else if (mulAdd) {
            auto value = operands[0];
            if (operation == "MulAddAfterAdd") {
                value = std::make_shared<ov::op::v1::Add>(value, ov::op::v0::Constant::create(type, ov::Shape{}, {2}));
            }
            result = std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(value, operands[1]),
                                                       operands[2]);
        } else {
            OPENVINO_THROW("Unexpected operation: ", operation);
        }
        function = std::make_shared<ov::Model>(ov::OutputVector{result}, parameters, "EltwiseI32Precision");
    }
};

TEST_P(EltwiseI32PrecisionTest, CompareWithRefs) {
    run();
    const bool snippets = std::get<3>(GetParam());
    CheckPluginRelatedResults(compiledModel, snippets ? "Subgraph" : "Eltwise");
    const bool referenceDivide =
        std::get<0>(GetParam()) == "DivideAfterAdd" && std::get<4>(GetParam()) == ov::element::i32;
    if (std::get<0>(GetParam()) == "DivideAfterAdd") {
        // Constant divisors can be folded or decomposed. A parameter divisor must retain the integer reference path.
        if (referenceDivide && !std::get<2>(GetParam())) {
            size_t referenceNodes = 0;
            size_t jitNodes = 0;
            for (const auto& node : compiledModel.get_runtime_model()->get_ops()) {
                const auto& info = node->get_rt_info();
                if (info.at(ov::exec_model_info::LAYER_TYPE).as<std::string>() == "Eltwise") {
                    const auto implementation = info.at(ov::exec_model_info::IMPL_TYPE).as<std::string>();
                    referenceNodes += implementation == "ref_i32";
                    jitNodes += implementation == "jit_i32";
                }
            }
            EXPECT_GE(referenceNodes, 1);
            EXPECT_GE(jitNodes, 1);
        }
        return;
    }
    // Sharing the parameter second input prevents generic Eltwise fusion for this chain.
    const bool unfusedChain = std::get<0>(GetParam()) == "ArithmeticChain" && !std::get<2>(GetParam()) && !snippets;
    CheckNumberOfNodesWithType(compiledModel, snippets ? "Subgraph" : "Eltwise", unfusedChain ? 2 : 1);
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Eltwise_I32Precision,
    EltwiseI32PrecisionTest,
    ::testing::Combine(::testing::Values("Add",
                                         "Subtract",
                                         "Multiply",
                                         "Minimum",
                                         "Maximum",
                                         "Abs",
                                         "Negative",
                                         "SquaredDifference",
                                         "Floor",
                                         "MulAdd",
                                         "MulAddAfterAdd",
                                         "ArithmeticChain"),
                       ::testing::Values(size_t{0}, size_t{1}, size_t{2}, size_t{3}, size_t{4}, size_t{5}),
                       ::testing::Bool(),
                       ::testing::Bool(),
                       ::testing::Values(ov::element::i32, ov::element::f32)),
    EltwiseI32PrecisionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Eltwise_I32Precision_Divide,
    EltwiseI32PrecisionTest,
    ::testing::Combine(::testing::Values("DivideAfterAdd"),
                       ::testing::Values(size_t{0}, size_t{1}, size_t{2}, size_t{3}, size_t{4}, size_t{5}),
                       ::testing::Bool(),
                       ::testing::Values(false),
                       ::testing::Values(ov::element::i32, ov::element::f32)),
    EltwiseI32PrecisionTest::getTestCaseName);

using EltwiseI32ConvertParams = std::tuple<ov::element::Type, size_t, size_t, bool, bool>;

class EltwiseI32ConvertTest : public testing::WithParamInterface<EltwiseI32ConvertParams>,
                              public SubgraphBaseTest,
                              public CPUTestUtils::CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<EltwiseI32ConvertParams>& info) {
        const auto& [sourcePrecision, shapeCase, convertedPort, disableSnippets, floor] = info.param;
        return std::string(floor ? "Floor_Source_" : "Multiply_Source_") + sourcePrecision.to_string() + "_Shape" +
               std::to_string(shapeCase) + "_ConvertedPort" + std::to_string(convertedPort) + "_DisableSnippets" +
               std::to_string(disableSnippets);
    }

protected:
    void generate_inputs(const std::vector<ov::Shape>& shapes) override {
        constexpr std::array<uint8_t, 8> unsignedValues = {255, 129, 128, 127, 0, 1, 2, 3};
        constexpr std::array<int8_t, 8> signedValues = {-128, -127, -1, 0, 1, 2, 126, 127};
        constexpr std::array<float, 8> floatValues = {255.75F, -127.75F, -1.75F, -0.75F, 0.75F, 1.75F, 126.5F, 127.5F};
        inputs.clear();
        const auto& parameters = function->get_parameters();
        for (size_t port = 0; port < parameters.size(); ++port) {
            const auto type = parameters[port]->get_element_type();
            ov::Tensor tensor(type, shapes[port]);
            for (size_t i = 0; i < tensor.get_size(); ++i) {
                if (type == ov::element::u8) {
                    tensor.data<uint8_t>()[i] = unsignedValues[i % unsignedValues.size()];
                } else if (type == ov::element::i8) {
                    tensor.data<int8_t>()[i] = signedValues[i % signedValues.size()];
                } else if (type == ov::element::f16) {
                    tensor.data<ov::float16>()[i] = ov::float16(floatValues[i % floatValues.size()]);
                } else if (type == ov::element::f32) {
                    tensor.data<float>()[i] = floatValues[i % floatValues.size()];
                } else {
                    // Odd products above 2^24 expose accidental conversion of the integer operand to f32.
                    tensor.data<int32_t>()[i] = (1 << 23) + 1;
                }
            }
            inputs.emplace(parameters[port], tensor);
        }
    }

    void SetUp() override {
        const auto& [sourcePrecision, shapeCase, convertedPort, disableSnippets, floor] = GetParam();
        std::vector<InputShape> shapes;
        switch (shapeCase) {
        case 0:
            shapes = {{{}, {{1, 1}}}, {{}, {{1, 1}}}};
            break;
        case 1:
            shapes = {{{}, {{1, 4}}}, {{}, {{1, 4}}}};
            break;
        case 2:
            shapes = {{{}, {{1, 29}}}, {{}, {{1, 29}}}};
            break;
        case 3:
            shapes = {{{}, {{1, 1}}}, {{}, {{2, 29}}}};
            break;
        case 4:
            // Rank seven keeps this graph outside default Snippets tokenization.
            shapes = {{{}, {{1, 1, 1, 1, 1, 1, 29}}}, {{}, {{1, 1, 1, 1, 1, 1, 29}}}};
            break;
        case 5:
            shapes = {{{-1, -1}, {{1, 1}, {1, 29}, {1, 5}}}, {{-1, -1}, {{2, 29}, {1, 29}, {1, 5}}}};
            break;
        default:
            OPENVINO_THROW("Unexpected shape case");
        }
        if (convertedPort == 1) {
            std::swap(shapes[0], shapes[1]);
        }
        if (floor) {
            shapes.resize(1);
        }
        init_input_shapes(shapes);
        targetDevice = ov::test::utils::DEVICE_CPU;
        abs_threshold = 0;
        rel_threshold = 0;
        // Generic Eltwise reports the first input's storage precision in its implementation name.
        selectedType = convertedPort == 0 && sourcePrecision.bitwidth() == 8 ? "jit_(I8|i32)" : "jit_i32";
        if (disableSnippets) {
            configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::DISABLE));
        }
        ov::ParameterVector parameters;
        ov::OutputVector operands;
        for (size_t port = 0; port < shapes.size(); ++port) {
            auto parameter =
                std::make_shared<ov::op::v0::Parameter>(port == convertedPort ? sourcePrecision : ov::element::i32,
                                                        inputDynamicShapes[port]);
            parameters.push_back(parameter);
            if (port == convertedPort) {
                operands.push_back(std::make_shared<ov::op::v0::Convert>(parameter, ov::element::i32));
            } else {
                operands.push_back(parameter);
            }
        }
        std::shared_ptr<ov::Node> result;
        if (floor) {
            result = std::make_shared<ov::op::v0::Floor>(operands[0]);
        } else {
            result = std::make_shared<ov::op::v1::Multiply>(operands[0], operands[1]);
        }
        function = std::make_shared<ov::Model>(ov::OutputVector{result}, parameters, "EltwiseI32Convert");
    }
};

TEST_P(EltwiseI32ConvertTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, std::set<std::string>{"Eltwise", "Subgraph"});
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Eltwise_I32Precision_Convert,
    EltwiseI32ConvertTest,
    ::testing::Combine(::testing::Values(ov::element::u8, ov::element::i8, ov::element::f16),
                       ::testing::Values(size_t{0}, size_t{1}, size_t{2}, size_t{3}, size_t{4}, size_t{5}),
                       ::testing::Values(size_t{0}, size_t{1}),
                       ::testing::Bool(),
                       ::testing::Values(false)),
    EltwiseI32ConvertTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Eltwise_I32Precision_ConvertFloor,
                         EltwiseI32ConvertTest,
                         ::testing::Combine(::testing::Values(ov::element::f16, ov::element::f32),
                                            ::testing::Values(size_t{0}, size_t{2}, size_t{4}, size_t{5}),
                                            ::testing::Values(size_t{0}),
                                            ::testing::Bool(),
                                            ::testing::Values(true)),
                         EltwiseI32ConvertTest::getTestCaseName);

}  // namespace
}  // namespace ov::test
#endif
