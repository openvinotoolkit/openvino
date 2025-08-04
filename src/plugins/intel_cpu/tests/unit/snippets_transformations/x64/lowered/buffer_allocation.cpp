// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset1.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/mark_loops.hpp"
#include "snippets/lowered/pass/init_loops.hpp"
#include "snippets/lowered/pass/insert_load_store.hpp"
#include "snippets/lowered/pass/insert_loops.hpp"
#include "snippets/lowered/pass/allocate_buffers.hpp"
#include "snippets/lowered/pass/fuse_loops.hpp"
#include "snippets/lowered/pass/split_loops.hpp"
#include "snippets/lowered/pass/insert_buffers.hpp"
#include "snippets/lowered/pass/reduce_decomposition.hpp"
#include "snippets/op/load.hpp"
#include "snippets/op/store.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "snippets/op/powerstatic.hpp"
#include "snippets/op/reduce.hpp"

#include "transformations/snippets/common/shape_inference.hpp"
#include "transformations/snippets/x64/pass/lowered/brgemm_cpu_blocking.hpp"
#include "transformations/snippets/x64/pass/lowered/insert_brgemm_copy_buffers.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "lowered/pass/buffer_allocation.hpp"


namespace ov {
namespace test {
namespace snippets {
using BrgemmConfig = intel_cpu::brgemm_utils::BrgemmConfig;

class BufferAllocationCPUTest : public BufferAllocationTest {
protected:
    std::shared_ptr<ov::snippets::IShapeInferSnippetsFactory> GetShapeInferFactory() const override {
        return std::make_shared<ov::snippets::CPUShapeInferSnippetsFactory>();
    }

    std::vector<ov::snippets::lowered::pass::PassPipeline::PositionedPassLowered> getBackendSpecificPasses() override {
        std::vector<ov::snippets::lowered::pass::PassPipeline::PositionedPassLowered> backend_passes;

        // Add BrgemmCPUBlocking after MarkLoops
        backend_passes.emplace_back(
            ov::snippets::pass::PassPosition(ov::snippets::pass::PassPosition::Place::After,
                                             ov::snippets::lowered::pass::MarkLoops::get_type_info_static()),
            std::make_shared<ov::intel_cpu::pass::BrgemmCPUBlocking>());

        // Add InsertBrgemmCopyBuffers after SplitLoops
        backend_passes.emplace_back(
            ov::snippets::pass::PassPosition(ov::snippets::pass::PassPosition::Place::After,
                                             ov::snippets::lowered::pass::SplitLoops::get_type_info_static()),
            std::make_shared<ov::intel_cpu::pass::InsertBrgemmCopyBuffers>());

        return backend_passes;
    }

    static void MarkOp(const std::shared_ptr<ov::Node>& node,
                       const std::vector<std::vector<size_t>>& in_subtensors,
                       const std::vector<std::vector<size_t>>& out_subtensors) {
        BufferAllocationTest::MarkOp(node, in_subtensors, out_subtensors);
    }
};

class MHAFP32BufferAllocationTest : public BufferAllocationCPUTest {
protected:
    std::shared_ptr<ov::Model> GetModel(const std::vector<ov::PartialShape>& shapes) const override {
        const auto subtensor_scalar = std::vector<size_t>{1};
        const auto subtensor_power = std::vector<size_t>{1, ov::snippets::utils::get_full_dim_value()};
        const auto subtensor_full = std::vector<size_t>(2, ov::snippets::utils::get_full_dim_value());

        const BrgemmConfig brgemm_config(dnnl::impl::cpu::x64::cpu_isa_t::avx512_core,
                                         ov::element::f32,
                                         ov::element::f32,
                                         ov::element::f32,
                                         false,
                                         false);

        // Dims are selected in order to have blocking loops by each dim
        OPENVINO_ASSERT(shapes.size() == 3, "Incorrect count of input shapes");
        const auto parameter0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shapes[0]);
        const auto parameter1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shapes[1]);
        const auto parameter2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shapes[2]);

        const auto order = std::vector<size_t>{0, 2, 3, 1};
        const auto load_reshape = std::make_shared<ov::snippets::op::LoadReorder>(parameter1, 1, 0, order);
        const auto store = std::make_shared<ov::snippets::op::Store>(load_reshape);
        const auto relu0 = std::make_shared<ov::op::v0::Relu>(store);
        const auto brgemm_cpu0 =
            std::make_shared<ov::intel_cpu::BrgemmCPU>(OutputVector{parameter0, relu0}, brgemm_config);

        const auto relu1 = std::make_shared<ov::op::v0::Relu>(brgemm_cpu0);

        // Decomposed Softmax
        const auto reduce_max = std::make_shared<ov::snippets::op::ReduceMax>(relu1, 3);
        ov::snippets::op::ReduceBase::compute_and_set_reduce_subtensors(reduce_max);
        const auto subtract = std::make_shared<ov::op::v1::Subtract>(relu1, reduce_max);
        const auto exp = std::make_shared<ov::op::v0::Exp>(subtract);

        const auto reduce_sum = std::make_shared<ov::snippets::op::ReduceSum>(exp, 3);
        ov::snippets::op::ReduceBase::compute_and_set_reduce_subtensors(reduce_sum);
        const auto power = std::make_shared<ov::snippets::op::PowerStatic>(reduce_sum, -1.f);
        const auto multiply = std::make_shared<ov::op::v1::Multiply>(exp, power);

        const auto brgemm_cpu1 =
            std::make_shared<ov::intel_cpu::BrgemmCPU>(OutputVector{multiply, parameter2}, brgemm_config);

        const auto relu2 = std::make_shared<ov::op::v0::Relu>(brgemm_cpu1);

        const auto body = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(relu2), ov::ParameterVector{parameter0, parameter1, parameter2});

        MarkOp(load_reshape, {subtensor_scalar}, {subtensor_scalar});
        MarkOp(store, {subtensor_scalar}, {subtensor_scalar});
        MarkOp(power, {subtensor_power}, {subtensor_power});

        MarkOp(brgemm_cpu0, {subtensor_full, subtensor_full}, {subtensor_full});
        MarkOp(brgemm_cpu1, {subtensor_full, subtensor_full}, {subtensor_full});

        ov::snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(load_reshape->input(0))->set_layout(order);

        return body;
    }
};

class MHABF16AMXBufferAllocationTest : public BufferAllocationCPUTest {
protected:
    std::shared_ptr<ov::Model> GetModel(const std::vector<ov::PartialShape>& shapes) const override {
        const auto subtensor_scalar = std::vector<size_t>{1};
        const auto subtensor_power = std::vector<size_t>{1, ov::snippets::utils::get_full_dim_value()};
        const auto subtensor_full = std::vector<size_t>(2, ov::snippets::utils::get_full_dim_value());
        const auto subtensor_flat = std::vector<size_t>(1, ov::snippets::utils::get_full_dim_value());

        const BrgemmConfig brgemm_config(dnnl::impl::cpu::x64::cpu_isa_t::avx512_core_amx,
                                         ov::element::bf16,
                                         ov::element::bf16,
                                         ov::element::bf16,
                                         false,
                                         false);

        OPENVINO_ASSERT(shapes.size() == 3, "Incorrect count of input shapes");
        const auto parameter0 = std::make_shared<ov::op::v0::Parameter>(ov::element::bf16, shapes[0]);
        const auto parameter1 = std::make_shared<ov::op::v0::Parameter>(ov::element::bf16, shapes[1]);
        const auto parameter2 = std::make_shared<ov::op::v0::Parameter>(ov::element::bf16, shapes[2]);

        const auto order = std::vector<size_t>{0, 2, 3, 1};
        const auto load_reshape = std::make_shared<ov::snippets::op::LoadReorder>(parameter1, 1, 0, order);
        const auto store = std::make_shared<ov::snippets::op::Store>(load_reshape);
        const auto convert0 = std::make_shared<ov::snippets::op::ConvertSaturation>(store, ov::element::f32);
        const auto relu0 = std::make_shared<ov::op::v0::Relu>(convert0);
        const auto convert1 = std::make_shared<ov::snippets::op::ConvertSaturation>(relu0, ov::element::bf16);

        const auto brgemm_copyb0 = std::make_shared<ov::intel_cpu::BrgemmCopyB>(convert1, brgemm_config);
        const auto scratch0 = std::make_shared<ov::snippets::op::Buffer>(ov::Shape{ov::intel_cpu::BrgemmCPU::SCRATCH_BYTE_SIZE});
        const auto brgemm_cpu0 =
            std::make_shared<ov::intel_cpu::BrgemmCPU>(OutputVector{parameter0, brgemm_copyb0->output(0), scratch0},
                                                       brgemm_config);

        const auto relu1 = std::make_shared<ov::op::v0::Relu>(brgemm_cpu0);

        // Decomposed Softmax
        const auto reduce_max = std::make_shared<ov::snippets::op::ReduceMax>(relu1, 3);
        ov::snippets::op::ReduceBase::compute_and_set_reduce_subtensors(reduce_max);
        const auto subtract = std::make_shared<ov::op::v1::Subtract>(relu1, reduce_max);
        const auto exp = std::make_shared<ov::op::v0::Exp>(subtract);

        const auto reduce_sum = std::make_shared<ov::snippets::op::ReduceSum>(exp, 3);
        ov::snippets::op::ReduceBase::compute_and_set_reduce_subtensors(reduce_sum);
        const auto power = std::make_shared<ov::snippets::op::PowerStatic>(reduce_sum, -1.f);
        const auto multiply = std::make_shared<ov::op::v1::Multiply>(exp, power);

        const auto convert2 = std::make_shared<ov::snippets::op::ConvertSaturation>(multiply, ov::element::bf16);

        const auto brgemm_copyb1 = std::make_shared<ov::intel_cpu::BrgemmCopyB>(parameter2, brgemm_config);
        const auto scratch1 = std::make_shared<ov::snippets::op::Buffer>(ov::Shape{ov::intel_cpu::BrgemmCPU::SCRATCH_BYTE_SIZE});
        const auto brgemm_cpu1 =
            std::make_shared<ov::intel_cpu::BrgemmCPU>(OutputVector{convert2, brgemm_copyb1->output(0), scratch1},
                                                       brgemm_config);

        const auto relu2 = std::make_shared<ov::op::v0::Relu>(brgemm_cpu1);

        const auto body = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(relu2), ov::ParameterVector{parameter0, parameter1, parameter2});

        MarkOp(load_reshape, {subtensor_scalar}, {subtensor_scalar});
        MarkOp(store, {subtensor_scalar}, {subtensor_scalar});
        MarkOp(power, {subtensor_power}, {subtensor_power});

        MarkOp(brgemm_cpu0, {subtensor_full, subtensor_full, subtensor_flat}, {subtensor_full});
        MarkOp(brgemm_cpu1, {subtensor_full, subtensor_full, subtensor_flat}, {subtensor_full});
        MarkOp(brgemm_copyb0, {subtensor_flat}, {subtensor_full});
        MarkOp(brgemm_copyb1, {subtensor_flat}, {subtensor_full});
        MarkOp(scratch0, {}, {subtensor_flat});
        MarkOp(scratch1, {}, {subtensor_flat});

        ov::snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(load_reshape->input(0))->set_layout(order);

        return body;
    }
};

TEST_P(MHAFP32BufferAllocationTest, BufferAllocationCPU) {
    Validate();
}

TEST_P(MHABF16AMXBufferAllocationTest, BufferAllocationCPU) {
    // Scratchpad memory for AMX with CopyA (dynamic case) has allocation size which depends on element count in vector register.
    // So the current `expected_allocation_size` in the test is targeted on real AVX512 platforms with vector registers with 512 bits.
    // If the test infrastructure has AVX2, the allocation size will not be matched.
    if (!with_cpu_x86_avx512_core())
        GTEST_SKIP();
    Validate();
}

namespace BufferAllocationCPUTest_Instances {

std::vector<ov::PartialShape> static_shapes = {
    { 1, 12, 1024, 1024 }, {1, 128, 12, 1024 }, {1, 12, 128, 256 },
};

std::vector<ov::PartialShape> dynamic_shapes = {
    { -1, -1, -1, -1 }, { -1, -1, -1, -1 }, { -1, -1, -1, -1 },
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHANotOptimizedWSplit, MHAFP32BufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(static_shapes),
                                 ::testing::Values(false),
                                 ::testing::Values(true),
                                 ::testing::Values(591360), // Each Buffer has own allocated memory
                                 ::testing::Values(7),     // Each Buffer has unique ID
                                 ::testing::Values(7)),    // Each Buffer has unique cluster ID
                         BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHAOptimizedWSplit, MHAFP32BufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(static_shapes),
                                 ::testing::Values(true),
                                 ::testing::Values(true),
                                 ::testing::Values(573440), // (Buffer before brgemm) + (between brgemms) + (after brgemm)
                                 ::testing::Values(2),     // (Buffer before brgemm0 and after brgemm1) + (between brgemms)
                                 ::testing::Values(3)),    // (Buffer before brgemm0) + (between brgemms) + (after brgemm1)
                         BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHANotOptimizedWOSplit, MHAFP32BufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(static_shapes),
                                 ::testing::Values(false),
                                 ::testing::Values(false),
                                 ::testing::Values(2622976), // Each Buffer has own allocated memory
                                 ::testing::Values(7),      // Each Buffer has unique ID
                                 ::testing::Values(7)),     // Each Buffer has unique cluster ID
                         BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHAOptimizedWOSplit, MHAFP32BufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(static_shapes),
                                 ::testing::Values(true),
                                 ::testing::Values(false),
                                 ::testing::Values(1572864), // (between brgemms) + (Buffer before brgemm0 and after brgemm1)
                                 ::testing::Values(2),     // (Buffer before brgemm0 and after brgemm1) + (between brgemms)
                                 ::testing::Values(3)),    // (Buffer before brgemm0) + (between brgemms) + (after brgemm1)
                         BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHABF16AMXNotOptimizedWSplit, MHABF16AMXBufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(static_shapes),
                                 ::testing::Values(false),
                                 ::testing::Values(true),
                                 ::testing::Values(713984),
                                 ::testing::Values(11),
                                 ::testing::Values(11)),
                         BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHABF16AMXOptimizedWSplit, MHABF16AMXBufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(static_shapes),
                                 ::testing::Values(true),
                                 ::testing::Values(true),
                                 ::testing::Values(524288),
                                 ::testing::Values(3),
                                 ::testing::Values(7)),
                         BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHABF16AMXNotOptimizedWOSplit, MHABF16AMXBufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(static_shapes),
                                 ::testing::Values(false),
                                 ::testing::Values(false),
                                 ::testing::Values(2491648),
                                 ::testing::Values(11),
                                 ::testing::Values(11)),
                         BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHABF16AMXOptimizedWOSplit, MHABF16AMXBufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(static_shapes),
                                 ::testing::Values(true),
                                 ::testing::Values(false),
                                 ::testing::Values(1671168),
                                 ::testing::Values(3),
                                 ::testing::Values(7)),
                         BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHAOptimizedWSplit_Dynamic, MHAFP32BufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(dynamic_shapes),
                                 ::testing::Values(true),
                                 ::testing::Values(true),
                                 ::testing::Values(0),  // no static clusters
                                 ::testing::Values(2),  // (Buffer before brgemm0 and after brgemm1) + (between brgemms)
                                 ::testing::Values(3)), // (Buffer before brgemm0) + (between brgemms) + (after brgemm1)
                         BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHABF16AMXOptimizedWSplit_Dynamic, MHABF16AMXBufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(dynamic_shapes),
                                 ::testing::Values(true),
                                 ::testing::Values(true),
                                 ::testing::Values(34816),  // only WSP buffers
                                 ::testing::Values(3),
                                 ::testing::Values(7)),
                         BufferAllocationTest::getTestCaseName);

}  // namespace BufferAllocationCPUTest_Instances
}  // namespace snippets
}  // namespace test
}  // namespace ov
