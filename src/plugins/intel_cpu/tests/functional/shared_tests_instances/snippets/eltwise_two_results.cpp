// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/eltwise_two_results.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {
namespace {

// todo: Remove the architecture constraint after isa sve_128 being supported. Because for now ARM Snippets only support isa asimd,
// but dnnl injector jit_uni_eltwise_injector_f32 requires isa being at least sve_128. So dnnl emitters used by these test cases
// are not supported for ARM architecture now.
#if defined(OPENVINO_ARCH_X86_64)
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise_TwoResults, EltwiseTwoResults,
                        ::testing::Combine(
                             ::testing::Values(InputShape {{}, {{1, 64, 10, 10}}}),
                             ::testing::Values(InputShape {{}, {{1, 64, 10,  1}}}),
                             ::testing::Values(2),
                             ::testing::Values(2),
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         EltwiseTwoResults::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise_TwoResults_Dynamic, EltwiseTwoResults,
                        ::testing::Combine(
                             ::testing::Values(InputShape {{-1, -1, -1, -1}, {{1, 64, 10, 10}, {2, 8, 2, 1}, {1, 64, 10, 10}}}),
                             ::testing::Values(InputShape {{{1, 2}, {1, 64}, {1, 10}, 1}, {{1, 64, 10, 1}, {2, 1, 1, 1}, {1, 64, 10, 1}}}),
                             ::testing::Values(2),
                             ::testing::Values(2),
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         EltwiseTwoResults::getTestCaseName);
#endif

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov