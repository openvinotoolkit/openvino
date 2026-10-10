// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/eltwise.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "nodes/kernels/riscv64/cpu_isa_traits.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Eltwise {
namespace {

/*
 * The motivation of this test is to validate different input and output precisions of Eltwise Op.
 * If IO data type is not supported by jit emitter, they should be converted
 * to supported types on operation inputs and outputs in JIT kernel
*/

static const std::vector<ov::test::utils::EltwiseTypes> ops() {
    // JIT us supported only when `gv` is available
    if (ov::intel_cpu::riscv64::mayiuse(ov::intel_cpu::riscv64::gv)) {
        return { utils::EltwiseTypes::ADD };
    }
    return {};
}

const std::vector<ElementType>& modNetTypes() {
    static const std::vector<ElementType> netTypes =
        ov::intel_cpu::riscv64::mayiuse(ov::intel_cpu::riscv64::gv)
            ? std::vector<ElementType>{ElementType::i32, ElementType::f32}
            : std::vector<ElementType>{};
    return netTypes;
}

const std::vector<ElementType>& modSnippetsNetTypes() {
    static const std::vector<ElementType> netTypes =
        modNetTypes().empty() ? std::vector<ElementType>{} : std::vector<ElementType>{ElementType::f32};
    return netTypes;
}

const std::vector<ov::AnyMap>& config_infer_prc_f32() {
    static const std::vector<ov::AnyMap> additionalConfig = {
        {{ov::hint::inference_precision.name(), ov::element::f32}},
    };
    return additionalConfig;
}

class EltwiseModNegativeCPUTest : public EltwiseLayerCPUTest {
protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            auto input = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
            const auto fillInput = [i, size = input.get_size()](auto* data) {
                for (size_t j = 0; j < size; ++j) {
                    const bool negativeDivisor = j % 2 == 0;
                    data[j] = i == 0 ? (negativeDivisor ? 5 : -5) : (negativeDivisor ? -3 : 3);
                }
            };
            if (funcInput.get_element_type() == ov::element::i32) {
                fillInput(input.data<int32_t>());
            } else if (funcInput.get_element_type() == ov::element::f32) {
                fillInput(input.data<float>());
            } else {
                FAIL() << "Unsupported MOD input precision: " << funcInput.get_element_type();
            }
            inputs.insert({funcInput.get_node_shared_ptr(), std::move(input)});
        }
    }
};

const std::vector<std::vector<ov::Shape>>& inputShapes() {
    static const std::vector<std::vector<ov::Shape>> inputShapes = {
        {{2, 4, 4, 1}},
        {{2, 4, 4, 128}},
        {{2, 4, 4, 131}},
        {{2, 17, 5, 4}},
        {{2, 19, 5, 4}, {1, 19, 1, 1}},
        {{2, 19, 5, 1}, {1, 19, 1, 4}},
    };
    return inputShapes;
}

const std::vector<std::vector<ov::Shape>>& modInputShapes() {
    static const std::vector<std::vector<ov::Shape>> inputShapes = {{{2, 4, 4, 1}}};
    return inputShapes;
}

auto makeModParams(const std::vector<ElementType>& netTypes, bool enforceSnippets) {
    return ::testing::Combine(
        ::testing::Combine(
            ::testing::ValuesIn(static_shapes_to_test_representation(modInputShapes())),
            ::testing::Values(utils::EltwiseTypes::MOD),
            ::testing::Values(utils::InputLayerType::PARAMETER),
            ::testing::Values(ov::test::utils::OpType::VECTOR),
            ::testing::ValuesIn(netTypes),
            ::testing::Values(ov::element::dynamic),
            ::testing::Values(ov::element::dynamic),
            ::testing::Values(ov::test::utils::DEVICE_CPU),
            ::testing::Values(ov::AnyMap{})),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D())),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(enforceSnippets));
}

const auto modParams = makeModParams(modNetTypes(), false);
const auto modSnippetsParams = makeModParams(modSnippetsNetTypes(), true);

TEST_P(EltwiseModNegativeCPUTest, CompareWithRefs) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_ModNegative, EltwiseModNegativeCPUTest, modParams, EltwiseLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ModNegativeSnippets,
                         EltwiseModNegativeCPUTest,
                         modSnippetsParams,
                         EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_jit = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes())),
                ::testing::ValuesIn(ops()),
                ::testing::ValuesIn(secondaryInputTypes()),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn({ ElementType::i8, ElementType::u8, ElementType::f16, ElementType::i32, ElementType::f32 }),
                ::testing::Values(ov::element::dynamic),
                ::testing::Values(ov::element::dynamic),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(config_infer_prc_f32())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D())),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_jit, EltwiseLayerCPUTest, params_4D_jit, EltwiseLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace Eltwise
}  // namespace test
}  // namespace ov
