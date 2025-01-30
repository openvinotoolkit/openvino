// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/eltwise.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Eltwise {
namespace {

const std::vector<ov::AnyMap>& config_infer_prc_f32() {
    static const std::vector<ov::AnyMap> additionalConfig = {
        {{ov::hint::inference_precision.name(), ov::element::f32}},
    };
    return additionalConfig;
}

const auto params_4D_jit = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D())),
                ::testing::ValuesIn({ utils::EltwiseTypes::ADD }),
                ::testing::ValuesIn(secondaryInputTypes()),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn({ ElementType::i8, ElementType::u8, ElementType::i32, ElementType::f32 }),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
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
