// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/src/classes/eltwise_chain.hpp"

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "custom/single_layer_tests/classes/comparison.hpp"
#include "internal_properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/general_utils.h"

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
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    EltwiseChainTest::getTestCaseName);

#if defined(OPENVINO_ARCH_ARM64)
// Scalar, full-vector, tail, broadcast and dynamic shapes without cross-products of unrelated parameters.
const std::vector<InputShape> tailArithmeticShape = {
    {{}, {{1, 29}}},
    {{}, {{1, 29}}},
    {{}, {{1, 29}}},
};
const std::vector<std::vector<InputShape>> arithmeticShapes = {
    {{{}, {{1, 1}}}, {{}, {{1, 1}}}, {{}, {{1, 1}}}},
    {{{}, {{1, 4}}}, {{}, {{1, 4}}}, {{}, {{1, 4}}}},
    tailArithmeticShape,
    {{{}, {{2, 29}}}, {{}, {{1, 1}}}, {{}, {{2, 29}}}},
    {{{-1, -1}, {{1, 29}, {3, 29}, {5, 1}, {2, 29}}},
     {{1, -1}, {{1, 29}, {1, 29}, {1, 1}, {1, 29}}},
     {{-1, -1}, {{1, 29}, {3, 29}, {5, 1}, {2, 29}}}},
};

const std::vector<std::vector<InputShape>> fourInputShapes = {
    {{{}, {{1, 29}}}, {{}, {{1, 29}}}, {{}, {{1, 29}}}, {{}, {{1, 29}}}},
};
const std::vector<std::vector<InputShape>> divideShapes = {tailArithmeticShape};
const std::vector<std::vector<InputShape>> convertShapes = {
    {{{}, {{1, 29}}}, {{}, {{1, 29}}}},
};
const std::vector<std::vector<InputShape>> floorShapes = {
    {{{}, {{1, 29}}}},
};

const std::vector<ElementType> threeI32Inputs(3, ElementType::i32);
const std::vector<ElementType> fourI32Inputs(4, ElementType::i32);
const std::vector<EltwiseChainOp> mulAdd{EltwiseTypes::MULTIPLY, EltwiseTypes::ADD};
const std::vector<EltwiseChainOp> mulAddAfterAdd{
    EltwiseTypes::ADD,
    EltwiseTypes::MULTIPLY,
    EltwiseTypes::ADD,
};
const std::vector<EltwiseChainOp> divideAfterAdd{EltwiseTypes::ADD, EltwiseTypes::DIVIDE};
const std::vector<EltwiseChainOp> comparisonChain{
    EltwiseTypes::ADD,
    EltwiseTypes::SUBTRACT,
    ComparisonTypes::EQUAL,
};
const std::vector<EltwiseChainOp> multiply{EltwiseTypes::MULTIPLY};
const std::vector<EltwiseChainOp> floor{ActivationTypes::Floor};
const std::vector<std::vector<ElementType>> convertPrecisions = {
    {ElementType::i8, ElementType::i32},
    {ElementType::u8, ElementType::i32},
    {ElementType::i32, ElementType::i8},
    {ElementType::i32, ElementType::u8},
};
const std::vector<std::vector<ElementType>> floorPrecisions = {
    {ElementType::f16},
    {ElementType::f32},
};

class EltwiseChainI32Test : public EltwiseChainTest, public CPUTestUtils::CPUTestsBase {
protected:
    void SetUp() override {
        EltwiseChainTest::SetUp();
        abs_threshold = 0;
        rel_threshold = 0;
        const auto& precisions = std::get<2>(this->GetParam());
        const auto& ops = std::get<3>(this->GetParam());
        const auto conversion = std::get<5>(this->GetParam());
        if (std::get<4>(this->GetParam()) == PostNode::LOGICAL_NOT) {
            selectedType = "jit_(i32|I8)";
        } else if (conversion == ov::element::i32 &&
                   ov::intel_cpu::any_of(precisions.front(), ElementType::i8, ElementType::u8)) {
            selectedType = "jit_(I8|i32)";
        } else if (std::holds_alternative<EltwiseTypes>(ops.back()) &&
                   std::get<EltwiseTypes>(ops.back()) == EltwiseTypes::DIVIDE) {
            selectedType = "(jit|ref)_i32";
        } else {
            selectedType = "jit_i32";
        }
    }

    void validate() override {
        SubgraphBaseTest::validate();
        CheckPluginRelatedResults(compiledModel, std::set<std::string>{"Eltwise", "Subgraph"});
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& inputPrecisions = std::get<2>(this->GetParam());
        const auto& ops = std::get<3>(this->GetParam());
        const auto conversion = std::get<5>(this->GetParam());
        const bool comparisonChain = std::holds_alternative<ComparisonTypes>(ops.back());
        const bool divideChain = std::holds_alternative<EltwiseTypes>(ops.back()) &&
                                 std::get<EltwiseTypes>(ops.back()) == EltwiseTypes::DIVIDE;
        const bool narrowIntegerConversion =
            conversion == ov::element::i32 && inputPrecisions.size() == 2 &&
            (ov::intel_cpu::any_of(inputPrecisions[0], ElementType::i8, ElementType::u8) ||
             ov::intel_cpu::any_of(inputPrecisions[1], ElementType::i8, ElementType::u8));
        const auto& window = *comparison::i32Windows().front();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            const auto type = funcInput.get_element_type();
            const auto seed = static_cast<int32_t>(i + 1);
            InputGenerateData data;
            if (type == ov::element::i8) {
                data = InputGenerateData(-128, 256, 1, seed);
            } else if (type == ov::element::u8) {
                data = InputGenerateData(0, 256, 1, seed);
            } else if (type.is_real()) {
                // Fractional inputs distinguish Convert-then-Floor from Floor-then-Convert for negative values.
                data = InputGenerateData(-127.75, 256, 4, seed);
            } else if (narrowIntegerConversion) {
                // A large exact i32 value whose product with any int8 value remains representable.
                data = InputGenerateData((1 << 23) + 1, 0, 1, seed);
            } else if (i == 0 || (comparisonChain && i + 1 == funcInputs.size())) {
                data = InputGenerateData(window.start_from, window.range, 1, seed);
            } else if (divideChain && i + 1 == funcInputs.size()) {
                // A divisor of two exposes precision lost by the preceding addition.
                data = InputGenerateData(2, 0, 1, seed);
            } else if (divideChain && i == 1) {
                data = InputGenerateData(1, 0, 1, seed);
            } else {
                // Keep intermediates in range.
                data = InputGenerateData(1, 2, 1, seed);
            }
            inputs.insert({funcInput.get_node_shared_ptr(),
                           ov::test::utils::create_and_fill_tensor(type, targetInputStaticShapes[i], data)});
        }
    }
};

class EltwiseChainI32SnippetsTest : public EltwiseChainI32Test {
protected:
    void SetUp() override {
        EltwiseChainI32Test::SetUp();
        configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::IGNORE_CALLBACK));
    }

    void validate() override {
        EltwiseChainI32Test::validate();
        CheckNumberOfNodesWithType(compiledModel, "Subgraph", 1);
        if (std::get<5>(this->GetParam()) != ov::element::dynamic) {
            // Snippets retains a Convert at the subgraph boundary; forced Eltwise execution checks full merging below.
            CheckNumberOfNodesWithType(compiledModel, "Convert", 1);
        }
    }
};

class EltwiseChainI32EltwiseTest : public EltwiseChainI32Test {
protected:
    void SetUp() override {
        EltwiseChainI32Test::SetUp();
        configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::DISABLE));
    }

    void validate() override {
        EltwiseChainI32Test::validate();
        if (std::get<5>(this->GetParam()) != ov::element::dynamic) {
            const auto& precisions = std::get<2>(this->GetParam());
            const bool fractionalInput = ov::intel_cpu::any_of(precisions.front(), ElementType::f16, ElementType::f32);
            CheckNumberOfNodesWithType(compiledModel, "Convert", fractionalInput ? 1 : 0);
            CheckNumberOfNodesWithType(compiledModel, "Eltwise", 1);
            return;
        }
        if (std::get<4>(this->GetParam()) == PostNode::LOGICAL_NOT) {
            CheckNumberOfNodesWithType(compiledModel, "Eltwise", 3);
            return;
        }

        const auto& ops = std::get<3>(this->GetParam());
        if (std::holds_alternative<EltwiseTypes>(ops.back()) &&
            std::get<EltwiseTypes>(ops.back()) == EltwiseTypes::DIVIDE) {
            CheckNumberOfNodesWithType(compiledModel, "Eltwise", 3);
            std::set<std::string> implementations;
            for (const auto& node : compiledModel.get_runtime_model()->get_ops()) {
                const auto& info = node->get_rt_info();
                if (info.count(ov::exec_model_info::IMPL_TYPE)) {
                    implementations.insert(info.at(ov::exec_model_info::IMPL_TYPE).as<std::string>());
                }
            }
            EXPECT_EQ(implementations.count("jit_i32"), 1);
            EXPECT_EQ(implementations.count("ref_i32"), 1);
            return;
        }
        CheckNumberOfNodesWithType(compiledModel, "Eltwise", 1);
    }
};

TEST_P(EltwiseChainI32SnippetsTest, CompareWithRefs) {
    run();
}

TEST_P(EltwiseChainI32EltwiseTest, CompareWithRefs) {
    run();
}

#define INSTANTIATE_I32_CHAIN(prefix, fixture, shapes, precisions, operations, post_node, conversion)                 \
    INSTANTIATE_TEST_SUITE_P(prefix,                                                                                  \
                             fixture,                                                                                 \
                             ::testing::Combine(::testing::ValuesIn(shapes),                                          \
                                                ::testing::Values(InputLayerType::PARAMETER),                         \
                                                ::testing::ValuesIn(precisions),                                      \
                                                ::testing::Values(operations),                                        \
                                                ::testing::Values(post_node),                                         \
                                                ::testing::Values(conversion),                                        \
                                                ::testing::Values(ov::test::utils::DEVICE_CPU)),                      \
                             EltwiseChainTest::getTestCaseName)

INSTANTIATE_I32_CHAIN(smoke_EltwiseChain_I32MulAdd_Snippets,
                      EltwiseChainI32SnippetsTest,
                      arithmeticShapes,
                      std::vector<std::vector<ElementType>>{threeI32Inputs},
                      mulAdd,
                      PostNode::EMPTY,
                      ov::element::dynamic);
INSTANTIATE_I32_CHAIN(smoke_EltwiseChain_I32MulAdd_Eltwise,
                      EltwiseChainI32EltwiseTest,
                      arithmeticShapes,
                      std::vector<std::vector<ElementType>>{threeI32Inputs},
                      mulAdd,
                      PostNode::EMPTY,
                      ov::element::dynamic);
INSTANTIATE_I32_CHAIN(smoke_EltwiseChain_I32FusedArithmetic_Snippets,
                      EltwiseChainI32SnippetsTest,
                      fourInputShapes,
                      std::vector<std::vector<ElementType>>{fourI32Inputs},
                      mulAddAfterAdd,
                      PostNode::EMPTY,
                      ov::element::dynamic);
INSTANTIATE_I32_CHAIN(smoke_EltwiseChain_I32FusedArithmetic_Eltwise,
                      EltwiseChainI32EltwiseTest,
                      fourInputShapes,
                      std::vector<std::vector<ElementType>>{fourI32Inputs},
                      mulAddAfterAdd,
                      PostNode::EMPTY,
                      ov::element::dynamic);
INSTANTIATE_I32_CHAIN(smoke_EltwiseChain_I32DivideFallback,
                      EltwiseChainI32EltwiseTest,
                      divideShapes,
                      std::vector<std::vector<ElementType>>{threeI32Inputs},
                      divideAfterAdd,
                      PostNode::EMPTY,
                      ov::element::dynamic);
INSTANTIATE_I32_CHAIN(smoke_EltwiseChain_I32ComparisonNot_Snippets,
                      EltwiseChainI32SnippetsTest,
                      fourInputShapes,
                      std::vector<std::vector<ElementType>>{fourI32Inputs},
                      comparisonChain,
                      PostNode::LOGICAL_NOT,
                      ov::element::dynamic);
INSTANTIATE_I32_CHAIN(smoke_EltwiseChain_I32ComparisonNot_Eltwise,
                      EltwiseChainI32EltwiseTest,
                      fourInputShapes,
                      std::vector<std::vector<ElementType>>{fourI32Inputs},
                      comparisonChain,
                      PostNode::LOGICAL_NOT,
                      ov::element::dynamic);
INSTANTIATE_I32_CHAIN(smoke_EltwiseChain_I32ConvertMerge_Snippets,
                      EltwiseChainI32SnippetsTest,
                      convertShapes,
                      convertPrecisions,
                      multiply,
                      PostNode::EMPTY,
                      ov::element::i32);
INSTANTIATE_I32_CHAIN(smoke_EltwiseChain_I32ConvertMerge_Eltwise,
                      EltwiseChainI32EltwiseTest,
                      convertShapes,
                      convertPrecisions,
                      multiply,
                      PostNode::EMPTY,
                      ov::element::i32);
INSTANTIATE_I32_CHAIN(smoke_EltwiseChain_I32FractionalFloor_Snippets,
                      EltwiseChainI32SnippetsTest,
                      floorShapes,
                      floorPrecisions,
                      floor,
                      PostNode::EMPTY,
                      ov::element::i32);
INSTANTIATE_I32_CHAIN(smoke_EltwiseChain_I32FractionalFloor_Eltwise,
                      EltwiseChainI32EltwiseTest,
                      floorShapes,
                      floorPrecisions,
                      floor,
                      PostNode::EMPTY,
                      ov::element::i32);

#undef INSTANTIATE_I32_CHAIN
#endif

}  // namespace
}  // namespace test
}  // namespace ov
