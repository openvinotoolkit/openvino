// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/identity.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace identity {

static const std::vector<ov::Shape> shapes = {
        {500},
        {4, 3, 210}
};

static const std::vector<ElementType> prc = {
        ElementType::f32,
        ElementType::f16,
        ElementType::bf16,
        ElementType::i32,
        ElementType::u16,
        ElementType::boolean
};

INSTANTIATE_TEST_SUITE_P(smoke_Param, IdentityLayerTestCPU,
        ::testing::Combine(
                ::testing::ValuesIn(shapes),
                ::testing::ValuesIn(prc),
                ::testing::Values(false),
                ::testing::Values(emptyCPUSpec),
                ::testing::Values(empty_plugin_config)),
        IdentityLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ParamConst, IdentityLayerTestCPU,
        ::testing::Combine(
                ::testing::ValuesIn(shapes),
                ::testing::ValuesIn(prc),
                ::testing::Values(true),
                ::testing::Values(emptyCPUSpec),
                ::testing::Values(empty_plugin_config)),
        IdentityLayerTestCPU::getTestCaseName);

}  // namespace identity
}  // namespace test
}  // namespace ov
