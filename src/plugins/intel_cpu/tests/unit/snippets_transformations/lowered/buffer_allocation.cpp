// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/mark_loops.hpp"
#include "snippets/lowered/pass/init_loops.hpp"
#include "snippets/lowered/pass/insert_load_store.hpp"
#include "snippets/lowered/pass/validate_loops.hpp"
#include "snippets/lowered/pass/insert_loops.hpp"
#include "snippets/lowered/pass/allocate_buffers.hpp"
#include "snippets/lowered/pass/fuse_loops.hpp"
#include "snippets/lowered/pass/split_loops.hpp"
#include "snippets/lowered/pass/insert_buffers.hpp"
#include "snippets/lowered/pass/softmax_decomposition.hpp"

#include "transformations/snippets/x64/shape_inference.hpp"
#include "transformations/snippets/x64/pass/lowered/brgemm_blocking.hpp"
#include "transformations/snippets/x64/pass/lowered/set_brgemm_copy_b_buffers_shape.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"


namespace ov {
namespace test {
namespace snippets {

/*  Note[74841]:
 *  This test is almost full copy of BufferAllocationTest class from openvino/src/common/snippets/tests/include/lowered/pass/buffer_allocation.hpp.
 *  The BufferAllocationTest class should be shared test class to reuse this structure in backend-specific tests in test infrastructure refactoring.
 */

typedef std::tuple<
    bool,   // Optimized pipeline
    bool,   // With SplitLoops opt
    size_t, // Expected Buffer size in bytes
    size_t  // Expected unique Buffer IDs count
> BufferAllocationCPUParams;

class BufferAllocationCPUTest : public testing::TestWithParam<BufferAllocationCPUParams> {
public:
    using VectorDims = ov::snippets::VectorDims;
    static std::string getTestCaseName(testing::TestParamInfo<BufferAllocationCPUParams> obj) {
        bool is_optimized, with_split_loops;
        size_t expected_size, expected_count;
        std::tie(is_optimized, with_split_loops, expected_size, expected_count) = obj.param;
        std::ostringstream result;
        result << "Opt=" << ov::test::utils::bool2str(is_optimized) << "_";
        result << "Split=" << ov::test::utils::bool2str(with_split_loops) << "_";
        result << "ExpBufferSize=" << expected_size << "_";
        result << "ExpBufferNum=" << expected_count;
        return result.str();
    }

protected:
    void SetUp() override {
        bool is_optimized, with_split_loops;
        std::tie(is_optimized, with_split_loops, m_expected_size, m_expected_count) = this->GetParam();

        const auto body = GetModel();
        m_linear_ir = ov::snippets::lowered::LinearIR(body, std::make_shared<ov::snippets::CPUShapeInferSnippetsFactory>());
        m_linear_ir.set_loop_depth(m_loop_depth);
        ApplyTransformations(is_optimized, with_split_loops);
    }

    void ApplyTransformations(bool is_optimized, bool with_split_loops) {
        ov::snippets::lowered::pass::PassPipeline pipeline;
        pipeline.register_pass<ov::intel_cpu::pass::BrgemmBlocking>();
        pipeline.register_pass<ov::snippets::lowered::pass::MarkLoops>(m_vector_size);
        pipeline.register_pass<ov::snippets::lowered::pass::SoftmaxDecomposition>(m_vector_size);
        pipeline.register_pass<ov::snippets::lowered::pass::FuseLoops>();
        if (with_split_loops)
            pipeline.register_pass<ov::snippets::lowered::pass::SplitLoops>();
        pipeline.register_pass<ov::snippets::lowered::pass::InsertBuffers>(2);
        pipeline.register_pass<ov::snippets::lowered::pass::InsertLoadStore>(m_vector_size);
        pipeline.register_pass<ov::snippets::lowered::pass::InitLoops>();
        pipeline.register_pass<ov::snippets::lowered::pass::InsertLoops>();
        pipeline.register_pass<ov::intel_cpu::pass::SetBrgemmCopyBBuffersShape>();
        pipeline.register_pass<ov::snippets::lowered::pass::AllocateBuffers>(m_buffer_scratchpad, is_optimized);
        pipeline.run(m_linear_ir);
    }

    void Validate() {
        std::set<size_t> gprs;
        for (const auto& expr : m_linear_ir) {
            if (const auto buffer = ov::as_type_ptr<ov::snippets::op::Buffer>(expr->get_node())) {
                gprs.insert(buffer->get_id());
            }
        }
        EXPECT_EQ(gprs.size(), m_expected_count);
        EXPECT_EQ(m_buffer_scratchpad, m_expected_size);
    }

    virtual std::shared_ptr<ov::Model> GetModel() const = 0;

    void MarkOp(const std::shared_ptr<ov::Node>& node, const std::vector<size_t>& subtensor) const {
        for (const auto& input : node->inputs())
            ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor_ptr(
                input, std::make_shared<ov::snippets::lowered::PortDescriptor>(input, subtensor));
        for (const auto& output : node->outputs())
            ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor_ptr(
                output, std::make_shared<ov::snippets::lowered::PortDescriptor>(output, subtensor));
    }

    size_t m_buffer_scratchpad = 0;
    ov::snippets::lowered::LinearIR m_linear_ir;

    size_t m_expected_size = 0;
    size_t m_expected_count = 0;

    size_t m_loop_depth = 2;
    size_t m_vector_size = 16;
};

class MHABF16AMXBufferAllocationTest : public BufferAllocationCPUTest {
protected:
    std::shared_ptr<ov::Model> GetModel() const override {
        const auto subtensor_scalar = std::vector<size_t>{1, 1};
        const auto subtensor_softmax = std::vector<size_t>{1, ov::snippets::lowered::PortDescriptor::ServiceDimensions::FULL_DIM};
        const auto subtensor_full = std::vector<size_t>(2, ov::snippets::lowered::PortDescriptor::ServiceDimensions::FULL_DIM);

        const auto parameter0 = std::make_shared<ov::op::v0::Parameter>(ov::element::bf16, ov::PartialShape({1, 12, 128, 64}));
        const auto parameter1 = std::make_shared<ov::op::v0::Parameter>(ov::element::bf16, ov::PartialShape({1, 128, 12, 64}));
        const auto parameter2 = std::make_shared<ov::op::v0::Parameter>(ov::element::bf16, ov::PartialShape({1, 12, 128, 64}));

        const auto load_reshape = std::make_shared<ov::snippets::op::LoadReshape>(parameter1, 1, 0, std::vector<size_t>{0, 2, 3, 1});
        const auto store = std::make_shared<ov::snippets::op::Store>(load_reshape);
        const auto convert0 = std::make_shared<ov::snippets::op::ConvertSaturation>(store, ov::element::f32);
        const auto relu0 = std::make_shared<ov::op::v0::Relu>(convert0);
        const auto convert1 = std::make_shared<ov::snippets::op::ConvertSaturation>(relu0, ov::element::bf16);

        const auto brgemm_copyb0 = std::make_shared<ov::intel_cpu::BrgemmCopyB>(
            convert1, ov::element::bf16, ov::intel_cpu::BrgemmCopyB::OnlyRepacking, 0, 0, 0);
        const auto scratch0 = std::make_shared<ov::snippets::op::Buffer>(ov::Shape{ov::intel_cpu::BrgemmCPU::SCRATCH_BYTE_SIZE});
        const auto brgemm_cpu0 = std::make_shared<ov::intel_cpu::BrgemmCPU>(
            parameter0, brgemm_copyb0->output(0), scratch0, ov::intel_cpu::BrgemmCPU::Type::AMX);
        brgemm_cpu0->set_m_block_size(32);

        const auto relu1 = std::make_shared<ov::op::v0::Relu>(brgemm_cpu0);
        const auto softmax = std::make_shared<ov::op::v1::Softmax>(relu1, 3);
        const auto convert2 = std::make_shared<ov::snippets::op::ConvertSaturation>(softmax, ov::element::bf16);

        const auto brgemm_copyb1 = std::make_shared<ov::intel_cpu::BrgemmCopyB>(
            parameter2, ov::element::bf16, ov::intel_cpu::BrgemmCopyB::OnlyRepacking, 0, 0, 0);
        const auto scratch1 = std::make_shared<ov::snippets::op::Buffer>(ov::Shape{ov::intel_cpu::BrgemmCPU::SCRATCH_BYTE_SIZE});
        const auto brgemm_cpu1 = std::make_shared<ov::intel_cpu::BrgemmCPU>(
            convert2, brgemm_copyb1->output(0), scratch1, ov::intel_cpu::BrgemmCPU::Type::AMX);
        brgemm_cpu1->set_m_block_size(32);

        const auto relu2 = std::make_shared<ov::op::v0::Relu>(brgemm_cpu1);

        const auto body = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(relu2), ov::ParameterVector{parameter0, parameter1, parameter2});

        MarkOp(load_reshape, subtensor_scalar);
        MarkOp(store, subtensor_scalar);
        MarkOp(softmax, subtensor_softmax);

        MarkOp(brgemm_cpu0, subtensor_full);
        MarkOp(brgemm_cpu1, subtensor_full);
        MarkOp(brgemm_copyb0, subtensor_full);
        MarkOp(brgemm_copyb1, subtensor_full);
        MarkOp(scratch0, subtensor_full);
        MarkOp(scratch1, subtensor_full);

        return body;
    }
};

TEST_P(MHABF16AMXBufferAllocationTest, BufferAllocationCPU) {
    Validate();
}


namespace BufferAllocationCPUTest_Instances {

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHABF16AMXNotOptimizedWSplit, MHABF16AMXBufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(false),
                                 ::testing::Values(true),
                                 ::testing::Values(196608),
                                 ::testing::Values(11)),
                         BufferAllocationCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHAOptimizedWSplit, MHABF16AMXBufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(true),
                                 ::testing::Values(true),
                                 ::testing::Values(90112),
                                 ::testing::Values(4)),
                         BufferAllocationCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHANotOptimizedWOSplit, MHABF16AMXBufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(false),
                                 ::testing::Values(false),
                                 ::testing::Values(393216),
                                 ::testing::Values(11)),
                         BufferAllocationCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHAOptimizedWOSplit, MHABF16AMXBufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(true),
                                 ::testing::Values(false),
                                 ::testing::Values(114688),
                                 ::testing::Values(4)),
                         BufferAllocationCPUTest::getTestCaseName);

}  // namespace BufferAllocationCPUTest_Instances
}  // namespace snippets
}  // namespace test
}  // namespace ov
