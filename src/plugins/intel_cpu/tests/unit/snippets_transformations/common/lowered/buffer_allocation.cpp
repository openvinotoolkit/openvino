// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"
#include "lowered/pass/buffer_allocation.hpp"

namespace ov {
namespace test {
namespace snippets {

class GenericBufferAllocationTest : public EltwiseBufferAllocationTest {};

TEST_P(GenericBufferAllocationTest, BufferAllocationCPU) {
    Validate();
}

namespace GenericBufferAllocationTest_Instances {

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_GenericNotOptimized,
                         GenericBufferAllocationTest,
                         ::testing::Combine(
                             ::testing::Values(std::vector<ov::PartialShape>{{1, 3, 100, 100}}),
                             ::testing::Values(false),
                             ::testing::Values(false),
                             ::testing::Values(80000),
                             ::testing::Values(2),
                             ::testing::Values(2)),
                         BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_GenericOptimized,
                         GenericBufferAllocationTest,
                         ::testing::Combine(
                             ::testing::Values(std::vector<ov::PartialShape>{{1, 3, 100, 100}}),
                             ::testing::Values(true),
                             ::testing::Values(false),
                             ::testing::Values(40000),
                             ::testing::Values(1),
                             ::testing::Values(1)),
                         BufferAllocationTest::getTestCaseName);

}  // namespace GenericBufferAllocationTest_Instances

}  // namespace snippets
}  // namespace test
}  // namespace ov
