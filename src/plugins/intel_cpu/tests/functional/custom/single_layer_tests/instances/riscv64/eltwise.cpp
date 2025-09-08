// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/eltwise.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "nodes/kernels/riscv64/cpu_isa_traits.hpp"

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

const std::vector<ov::AnyMap>& config_infer_prc_f32() {
    static const std::vector<ov::AnyMap> additionalConfig = {
        {{ov::hint::inference_precision.name(), ov::element::f32}},
    };
    return additionalConfig;
}

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
