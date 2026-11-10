// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <set>

#include "custom/single_layer_tests/classes/eltwise.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "utils/general_utils.h"
#if defined(OPENVINO_ARCH_ARM64)
#include "custom/single_layer_tests/classes/comparison.hpp"
#endif

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Eltwise {
namespace {

const std::vector<std::vector<InputShape>> bitwise_in_shapes_4D = {
    {{{1, -1, -1, -1}, {{1, 3, 2, 2}, {1, 3, 1, 1}}}, {{1, 3, 2, 2}, {{1, 3, 2, 2}}}},
    {{{1, -1, -1, -1}, {{1, 64, 2, 2}, {1, 64, 1, 1}}}, {{1, 64, 2, 2}, {{1, 64, 2, 2}}}},
};

const std::vector<CPUSpecificParams>& bitwiseSnippetsCpuParams() {
    static const std::vector<CPUSpecificParams> params = {CPUSpecificParams({}, {}, {}, "jit")};
    return params;
}

const auto params_4D_bitwise_snippets =
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(bitwise_in_shapes_4D),
                                          ::testing::ValuesIn({utils::EltwiseTypes::BITWISE_AND,
                                                               utils::EltwiseTypes::BITWISE_OR,
                                                               utils::EltwiseTypes::BITWISE_XOR}),
                                          ::testing::ValuesIn(secondaryInputTypes()),
                                          ::testing::ValuesIn({utils::OpType::VECTOR}),
                                          ::testing::ValuesIn({ElementType::i8, ElementType::u8}),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU),
                                          ::testing::Values(ov::AnyMap())),
                       ::testing::ValuesIn(bitwiseSnippetsCpuParams()),
                       ::testing::Values(emptyFusingSpec),
                       ::testing::Values(true));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Bitwise_Snippets,
                         EltwiseLayerCPUTest,
                         params_4D_bitwise_snippets,
                         EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_bitwise_scalar_snippets =
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(bitwise_in_shapes_4D),
                                          ::testing::ValuesIn({utils::EltwiseTypes::BITWISE_AND,
                                                               utils::EltwiseTypes::BITWISE_OR,
                                                               utils::EltwiseTypes::BITWISE_XOR}),
                                          ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                          ::testing::Values(utils::OpType::SCALAR),
                                          ::testing::ValuesIn({ElementType::i8, ElementType::u8}),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU),
                                          ::testing::Values(ov::AnyMap())),
                       ::testing::ValuesIn(bitwiseSnippetsCpuParams()),
                       ::testing::Values(emptyFusingSpec),
                       ::testing::Values(true));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Bitwise_Scalar_Snippets,
                         EltwiseLayerCPUTest,
                         params_4D_bitwise_scalar_snippets,
                         EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_bitwise_NOT_snippets =
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(bitwise_in_shapes_4D),
                                          ::testing::ValuesIn({utils::EltwiseTypes::BITWISE_NOT}),
                                          ::testing::ValuesIn({ov::test::utils::InputLayerType::CONSTANT}),
                                          ::testing::ValuesIn({utils::OpType::VECTOR}),
                                          ::testing::ValuesIn({ElementType::i8, ElementType::u8}),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU),
                                          ::testing::Values(ov::AnyMap())),
                       ::testing::ValuesIn(bitwiseSnippetsCpuParams()),
                       ::testing::Values(emptyFusingSpec),
                       ::testing::Values(true));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Bitwise_NOT_Snippets,
                         EltwiseLayerCPUTest,
                         params_4D_bitwise_NOT_snippets,
                         EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_int_jit = ::testing::Combine(
    ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D())),
        ::testing::ValuesIn({utils::EltwiseTypes::ADD, utils::EltwiseTypes::MULTIPLY}),
        ::testing::ValuesIn(secondaryInputTypes()),
        ::testing::ValuesIn(opTypes()),
        ::testing::ValuesIn({ElementType::i8, ElementType::u8, ElementType::f16, ElementType::i32, ElementType::f32}),
        ::testing::Values(ov::element::dynamic),
        ::testing::Values(ov::element::dynamic),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(additional_config())),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D())),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_int_jit, EltwiseLayerCPUTest, params_4D_int_jit, EltwiseLayerCPUTest::getTestCaseName);

#if defined(OPENVINO_ARCH_ARM64)
// Keep the existing Eltwise parameter tuple and use typed operation parameters to select safe exact windows.
ov::test::utils::InputGenerateData i32ArithmeticInput(utils::EltwiseTypes op, size_t inputIndex, int32_t seed) {
    size_t windowIndex = 0;
    switch (op) {
    case utils::EltwiseTypes::SUBTRACT:
        windowIndex = 1;
        break;
    case utils::EltwiseTypes::SQUARED_DIFF:
        windowIndex = 2;
        break;
    default:
        break;
    }

    // Multiplication uses a small nonzero RHS to keep the result representable.
    if (inputIndex != 0 && op == utils::EltwiseTypes::MULTIPLY) {
        return {1, 2, 1, seed};
    }
    const auto& window = *comparison::i32Windows().at(windowIndex);
    return {window.start_from, window.range, 1, seed};
}

class EltwiseLayerI32CPUTest : public EltwiseLayerCPUTest {
protected:
    ov::Tensor generate_eltwise_input(const ov::element::Type& type,
                                      const ov::Shape& shape,
                                      const bool adopt_intervals = false) override {
        if (type != ov::element::i32) {
            return EltwiseLayerCPUTest::generate_eltwise_input(type, shape, adopt_intervals);
        }
        const auto op = std::get<1>(std::get<0>(this->GetParam()));
        return ov::test::utils::create_and_fill_tensor(type, shape, i32ArithmeticInput(op, 1, 2));
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        EltwiseLayerCPUTest::generate_inputs(targetInputStaticShapes);
        const auto op = std::get<1>(std::get<0>(this->GetParam()));
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            if (funcInput.get_element_type() == ov::element::i32) {
                inputs[funcInput.get_node_shared_ptr()] = ov::test::utils::create_and_fill_tensor(
                    funcInput.get_element_type(),
                    targetInputStaticShapes[i],
                    i32ArithmeticInput(op, i, static_cast<int32_t>(i + 1)));
            }
        }
    }
};

TEST_P(EltwiseLayerI32CPUTest, CompareWithRefs) {
    run();
    const bool enforceSnippets = std::get<3>(this->GetParam());
    CheckNumberOfNodesWithType(compiledModel, enforceSnippets ? "Subgraph" : "Eltwise", 1);
    CheckPluginRelatedResults(compiledModel, std::set<std::string>{"Eltwise", "Subgraph"});
}

const std::vector<std::vector<InputShape>> i32ArithmeticShapes = {
    {{{}, {{1, 1}}}},
    {{{}, {{1, 4}}}},
    {{{}, {{1, 29}}}},
    {{{}, {{2, 29}}}, {{}, {{1, 1}}}},
};

const CPUSpecificParams i32ArithmeticCpuParams({}, {}, {}, "jit");

const std::vector<utils::EltwiseTypes> i32ArithmeticOps = {
    utils::EltwiseTypes::ADD,
    utils::EltwiseTypes::MULTIPLY,
    utils::EltwiseTypes::SUBTRACT,
    utils::EltwiseTypes::SQUARED_DIFF,
};

const auto i32ArithmeticParams = ::testing::Combine(
    ::testing::Combine(::testing::ValuesIn(i32ArithmeticShapes),
                       ::testing::ValuesIn(i32ArithmeticOps),
                       ::testing::Values(utils::InputLayerType::PARAMETER),
                       ::testing::Values(utils::OpType::VECTOR),
                       ::testing::Values(ElementType::i32),
                       ::testing::Values(ElementType::dynamic),
                       ::testing::Values(ElementType::dynamic),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(ov::AnyMap())),
    ::testing::Values(i32ArithmeticCpuParams),
    ::testing::Values(emptyFusingSpec),
    ::testing::ValuesIn(enforceSnippets()));

INSTANTIATE_TEST_SUITE_P(smoke_Eltwise_I32Exact_ARM64,
                         EltwiseLayerI32CPUTest,
                         i32ArithmeticParams,
                         EltwiseLayerI32CPUTest::getTestCaseName);

const auto i32ArithmeticDynamicParams = ::testing::Combine(
    ::testing::Combine(::testing::Values(inShapes_4D_dyn_param()),
                       ::testing::ValuesIn(i32ArithmeticOps),
                       ::testing::Values(utils::InputLayerType::PARAMETER),
                       ::testing::Values(utils::OpType::VECTOR),
                       ::testing::Values(ElementType::i32),
                       ::testing::Values(ElementType::dynamic),
                       ::testing::Values(ElementType::dynamic),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(ov::AnyMap())),
    ::testing::Values(i32ArithmeticCpuParams),
    ::testing::Values(emptyFusingSpec),
    ::testing::ValuesIn(enforceSnippets()));

INSTANTIATE_TEST_SUITE_P(smoke_Eltwise_I32Exact_Dynamic_ARM64,
                         EltwiseLayerI32CPUTest,
                         i32ArithmeticDynamicParams,
                         EltwiseLayerI32CPUTest::getTestCaseName);

// Parameter-only cases above cover the exact computation; a small constant suite covers constant folding and fusion.
const auto i32ArithmeticConstantParams = ::testing::Combine(
    ::testing::Combine(::testing::Values(i32ArithmeticShapes[2]),
                       ::testing::ValuesIn({utils::EltwiseTypes::ADD, utils::EltwiseTypes::SQUARED_DIFF}),
                       ::testing::Values(utils::InputLayerType::CONSTANT),
                       ::testing::Values(utils::OpType::VECTOR),
                       ::testing::Values(ElementType::i32),
                       ::testing::Values(ElementType::dynamic),
                       ::testing::Values(ElementType::dynamic),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(ov::AnyMap())),
    ::testing::Values(i32ArithmeticCpuParams),
    ::testing::Values(emptyFusingSpec),
    ::testing::ValuesIn(enforceSnippets()));

INSTANTIATE_TEST_SUITE_P(smoke_Eltwise_I32Exact_Constant_ARM64,
                         EltwiseLayerI32CPUTest,
                         i32ArithmeticConstantParams,
                         EltwiseLayerI32CPUTest::getTestCaseName);

// Safe near-limit windows: the result and every intermediate stay inside i32 while inputs lose f32 exactness.
class EltwiseLayerI32LimitCPUTest : public EltwiseLayerI32CPUTest {
protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto op = std::get<1>(std::get<0>(this->GetParam()));
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::test::utils::InputGenerateData data;
            switch (op) {
            case utils::EltwiseTypes::ADD:
                data = i == 0 ? ov::test::utils::InputGenerateData(std::numeric_limits<int32_t>::max() - 4.0, 3)
                              : ov::test::utils::InputGenerateData(-2, 1);
                break;
            case utils::EltwiseTypes::SUBTRACT:
                data = i == 0 ? ov::test::utils::InputGenerateData(std::numeric_limits<int32_t>::min() + 4.0, 3)
                              : ov::test::utils::InputGenerateData(1, 1);
                break;
            case utils::EltwiseTypes::MULTIPLY:
                data = i == 0 ? ov::test::utils::InputGenerateData(std::numeric_limits<int32_t>::max() / 2 - 2.0, 3)
                              : ov::test::utils::InputGenerateData(2, 1);
                break;
            default:
                data = i == 0 ? ov::test::utils::InputGenerateData(std::numeric_limits<int32_t>::max() - 4.0, 3, 1, 1)
                              : ov::test::utils::InputGenerateData(std::numeric_limits<int32_t>::max() - 5.0, 3, 1, 2);
                break;
            }
            inputs.insert({funcInput.get_node_shared_ptr(),
                           ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                   targetInputStaticShapes[i],
                                                                   data)});
        }
    }
};

TEST_P(EltwiseLayerI32LimitCPUTest, CompareWithRefs) {
    run();
    const bool enforceSnippets = std::get<3>(this->GetParam());
    CheckNumberOfNodesWithType(compiledModel, enforceSnippets ? "Subgraph" : "Eltwise", 1);
    CheckPluginRelatedResults(compiledModel, std::set<std::string>{"Eltwise", "Subgraph"});
}

const auto i32ArithmeticLimitParams = ::testing::Combine(
    ::testing::Combine(::testing::Values(i32ArithmeticShapes[2]),
                       ::testing::ValuesIn(i32ArithmeticOps),
                       ::testing::Values(utils::InputLayerType::PARAMETER),
                       ::testing::Values(utils::OpType::VECTOR),
                       ::testing::Values(ElementType::i32),
                       ::testing::Values(ElementType::dynamic),
                       ::testing::Values(ElementType::dynamic),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(ov::AnyMap())),
    ::testing::Values(i32ArithmeticCpuParams),
    ::testing::Values(emptyFusingSpec),
    ::testing::ValuesIn(enforceSnippets()));

INSTANTIATE_TEST_SUITE_P(smoke_Eltwise_I32Limit_ARM64,
                         EltwiseLayerI32LimitCPUTest,
                         i32ArithmeticLimitParams,
                         EltwiseLayerI32LimitCPUTest::getTestCaseName);
#endif

}  // namespace
}  // namespace Eltwise
}  // namespace test
}  // namespace ov
