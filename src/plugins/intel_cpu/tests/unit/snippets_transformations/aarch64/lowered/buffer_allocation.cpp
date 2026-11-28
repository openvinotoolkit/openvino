// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../../common/lowered/buffer_allocation_cpu.hpp"
#include "snippets/lowered/pass/mark_loops.hpp"
#include "snippets/lowered/pass/split_loops.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"
#include "transformations/snippets/aarch64/pass/lowered/gemm_cpu_blocking.hpp"
#include "transformations/snippets/aarch64/pass/lowered/insert_gemm_copy_buffers.hpp"

namespace ov {
namespace test {
namespace snippets {

class Aarch64BufferAllocationCPUTest : public BufferAllocationCPUTest {
protected:
    std::vector<ov::snippets::lowered::pass::PassPipeline::PositionedPassLowered> getBackendSpecificPasses() override {
        return {
            {ov::snippets::pass::PassPosition(ov::snippets::pass::PassPosition::Place::After,
                                              ov::snippets::lowered::pass::MarkLoops::get_type_info_static()),
             std::make_shared<ov::intel_cpu::pass::GemmCPUBlocking>()},
            {ov::snippets::pass::PassPosition(ov::snippets::pass::PassPosition::Place::After,
                                              ov::snippets::lowered::pass::SplitLoops::get_type_info_static()),
             std::make_shared<ov::intel_cpu::pass::aarch64::InsertGemmCopyBuffers>()},
        };
    }
};

class MHAFP32BufferAllocationTest : public MHAFP32BufferAllocationCPUTest<Aarch64BufferAllocationCPUTest> {
protected:
    std::shared_ptr<ov::Node> create_matmul(const std::shared_ptr<ov::Node>& lhs,
                                            const std::shared_ptr<ov::Node>& rhs) const override {
        using PortDescriptor = ov::snippets::modifier::MemoryAccess::PortDescriptor;
        return std::make_shared<ov::intel_cpu::aarch64::GemmCPU>(lhs,
                                                                 rhs,
                                                                 PortDescriptor{},
                                                                 PortDescriptor{},
                                                                 PortDescriptor{});
    }
};

TEST_P(MHAFP32BufferAllocationTest, BufferAllocationCPU) {
    Validate();
}

namespace BufferAllocationCPUTest_Instances {

// This FP32 graph has no GemmCopyB nodes. Its buffer lifetimes match the x64 Brgemm graph.

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHANotOptimizedWSplit,
                         MHAFP32BufferAllocationTest,
                         ::testing::Combine(::testing::Values(static_shapes),
                                            ::testing::Values(false),
                                            ::testing::Values(true),
                                            ::testing::Values(566784),  // Each Buffer has own allocated memory
                                            ::testing::Values(7),       // Each Buffer has unique ID
                                            ::testing::Values(7)),      // Each Buffer has unique cluster ID
                         BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_BufferAllocation_MHAOptimizedWSplit,
    MHAFP32BufferAllocationTest,
    ::testing::Combine(::testing::Values(static_shapes),
                       ::testing::Values(true),
                       ::testing::Values(true),
                       ::testing::Values(548864),  // (Buffer before gemm) + (between gemms) + (after gemm)
                       ::testing::Values(2),       // (Buffer before gemm0 and after gemm1) + (between gemms)
                       ::testing::Values(3)),      // (Buffer before gemm0) + (between gemms) + (after gemm1)
    BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHANotOptimizedWOSplit,
                         MHAFP32BufferAllocationTest,
                         ::testing::Combine(::testing::Values(static_shapes),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(2622976),  // Each Buffer has own allocated memory
                                            ::testing::Values(7),        // Each Buffer has unique ID
                                            ::testing::Values(7)),       // Each Buffer has unique cluster ID
                         BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_BufferAllocation_MHAOptimizedWOSplit,
    MHAFP32BufferAllocationTest,
    ::testing::Combine(::testing::Values(static_shapes),
                       ::testing::Values(true),
                       ::testing::Values(false),
                       ::testing::Values(1572864),  // (between gemms) + (Buffer before gemm0 and after gemm1)
                       ::testing::Values(2),        // (Buffer before gemm0 and after gemm1) + (between gemms)
                       ::testing::Values(3)),       // (Buffer before gemm0) + (between gemms) + (after gemm1)
    BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_BufferAllocation_MHAOptimizedWSplit_Dynamic,
    MHAFP32BufferAllocationTest,
    ::testing::Combine(::testing::Values(dynamic_shapes),
                       ::testing::Values(true),
                       ::testing::Values(true),
                       ::testing::Values(0),   // no static clusters
                       ::testing::Values(2),   // (Buffer before gemm0 and after gemm1) + (between gemms)
                       ::testing::Values(3)),  // (Buffer before gemm0) + (between gemms) + (after gemm1)
    BufferAllocationTest::getTestCaseName);

}  // namespace BufferAllocationCPUTest_Instances

}  // namespace snippets
}  // namespace test
}  // namespace ov
