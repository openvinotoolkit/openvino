// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/eltwise.hpp"

#include <cstdint>
#include <limits>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "utils/fusing_test_utils.hpp"

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
class EltwiseLayerI32CPUTest : public EltwiseLayerCPUTest {
protected:
    void SetUp() override {
        EltwiseLayerCPUTest::SetUp();
        abs_threshold = rel_threshold = 0;
    }

    void validate() override {
        EltwiseLayerCPUTest::validate();
        CheckNumberOfNodesWithType(compiledModel, std::get<3>(GetParam()) ? "Subgraph" : "Eltwise", 1);
    }

    ov::Tensor generate_eltwise_input(const ov::element::Type& type,
                                      const ov::Shape& shape,
                                      const bool /*adopt_intervals*/) override {
        return utils::create_and_fill_tensor(type, shape, input_data(1));
    }

    void generate_inputs(const std::vector<ov::Shape>& shapes) override {
        inputs.clear();
        const auto& parameters = function->get_parameters();
        for (size_t i = 0; i < parameters.size(); ++i) {
            inputs.emplace(parameters[i], utils::create_and_fill_tensor(ElementType::i32, shapes[i], input_data(i)));
        }
    }

private:
    utils::InputGenerateData input_data(size_t port) const {
        const auto op = std::get<1>(std::get<0>(GetParam()));
        const auto seed = static_cast<int32_t>(port + 1);
        // Choose large operands whose intermediate and final results stay in range.
        switch (op) {
        case utils::EltwiseTypes::MULTIPLY:
            return port == 0 ? utils::InputGenerateData(std::numeric_limits<int32_t>::max() / 2 - 2.0, 3, 1, seed)
                             : utils::InputGenerateData(2, 0, 1, seed);
        case utils::EltwiseTypes::SUBTRACT:
            return {-(1 << 24) - 1, 3, 1, seed};
        case utils::EltwiseTypes::SQUARED_DIFF:
            return {std::numeric_limits<int32_t>::max() - 2.0, 3, 1, seed};
        default:
            return {(1 << 24) - 1, 3, 1, seed};
        }
    }
};

TEST_P(EltwiseLayerI32CPUTest, CompareWithRefs) {
    run();
}

const std::vector<InputShape> i32TailShape{{{}, {{1, 29}}}};
const std::vector<std::vector<InputShape>> i32ArithmeticShapes = {
    {{{}, {{1, 1}}}},
    {{{}, {{1, 4}}}},
    i32TailShape,
    {{{}, {{2, 29}}}, {{}, {{1, 1}}}},
    inShapes_4D_dyn_param(),
};

auto i32_arithmetic_params(const std::vector<std::vector<InputShape>>& shapes,
                           const std::vector<utils::EltwiseTypes>& ops,
                           utils::InputLayerType inputType) {
    return ::testing::Combine(::testing::Combine(::testing::ValuesIn(shapes),
                                                 ::testing::ValuesIn(ops),
                                                 ::testing::Values(inputType),
                                                 ::testing::Values(utils::OpType::VECTOR),
                                                 ::testing::Values(ElementType::i32),
                                                 ::testing::Values(ElementType::dynamic),
                                                 ::testing::Values(ElementType::dynamic),
                                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                 ::testing::Values(ov::AnyMap())),
                              ::testing::Values(CPUSpecificParams({}, {}, {}, "jit")),
                              ::testing::Values(emptyFusingSpec),
                              ::testing::Bool());
}

INSTANTIATE_TEST_SUITE_P(smoke_Eltwise_I32Exact_ARM64,
                         EltwiseLayerI32CPUTest,
                         i32_arithmetic_params(i32ArithmeticShapes,
                                               {utils::EltwiseTypes::ADD,
                                                utils::EltwiseTypes::MULTIPLY,
                                                utils::EltwiseTypes::SUBTRACT,
                                                utils::EltwiseTypes::SQUARED_DIFF},
                                               utils::InputLayerType::PARAMETER),
                         EltwiseLayerI32CPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Eltwise_I32Exact_Constant_ARM64,
                         EltwiseLayerI32CPUTest,
                         i32_arithmetic_params({i32TailShape},
                                               {utils::EltwiseTypes::ADD, utils::EltwiseTypes::SQUARED_DIFF},
                                               utils::InputLayerType::CONSTANT),
                         EltwiseLayerI32CPUTest::getTestCaseName);
#endif

}  // namespace
}  // namespace Eltwise
}  // namespace test
}  // namespace ov
