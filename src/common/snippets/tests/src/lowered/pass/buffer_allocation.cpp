// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lowered/pass/buffer_allocation.hpp"

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
#include "snippets/lowered/pass/reduce_decomposition.hpp"

#include "common_test_utils/common_utils.hpp"


namespace ov {
namespace test {
namespace snippets {

std::string BufferAllocationTest::getTestCaseName(testing::TestParamInfo<ov::test::snippets::BufferAllocationParams> obj) {
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

void BufferAllocationTest::SetUp() {
    std::tie(m_is_buffer_optimized, m_with_split_loops, m_expected_size, m_expected_count) = this->GetParam();

    const auto body = GetModel();
    m_linear_ir = ov::snippets::lowered::LinearIR(body, std::make_shared<ov::snippets::IShapeInferSnippetsFactory>());
    m_linear_ir.set_loop_depth(m_loop_depth);
    // When Subgraph::control_flow_transformations become public method,
    // please use this method instead of ApplyTransformations
    ApplyTransformations(GetPassConfig());
}

std::shared_ptr<ov::snippets::lowered::pass::PassConfig> BufferAllocationTest::GetPassConfig() {
    auto config = std::make_shared<ov::snippets::lowered::pass::PassConfig>();
    if (!m_with_split_loops)
        config->disable<ov::snippets::lowered::pass::SplitLoops>();
    return config;
}

void BufferAllocationTest::MarkOp(const std::shared_ptr<ov::Node>& node, const std::vector<size_t>& subtensor) {
    for (const auto& input : node->inputs())
        ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor_ptr(
            input, std::make_shared<ov::snippets::lowered::PortDescriptor>(input, subtensor));
    for (const auto& output : node->outputs())
        ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor_ptr(
            output, std::make_shared<ov::snippets::lowered::PortDescriptor>(output, subtensor));
}

void BufferAllocationTest::ApplyTransformations(const std::shared_ptr<ov::snippets::lowered::pass::PassConfig>& pass_config) {
    ov::snippets::lowered::pass::PassPipeline pipeline(pass_config);
    pipeline.register_pass<ov::snippets::lowered::pass::MarkLoops>(m_vector_size);
    pipeline.register_pass<ov::snippets::lowered::pass::ReduceDecomposition>(m_vector_size);
    pipeline.register_pass<ov::snippets::lowered::pass::FuseLoops>();
    pipeline.register_pass<ov::snippets::lowered::pass::SplitLoops>();
    pipeline.register_pass<ov::snippets::lowered::pass::InsertBuffers>(2);
    pipeline.register_pass<ov::snippets::lowered::pass::InsertLoadStore>(m_vector_size);
    pipeline.register_pass<ov::snippets::lowered::pass::InitLoops>();
    pipeline.register_pass<ov::snippets::lowered::pass::InsertLoops>();
    pipeline.register_pass<ov::snippets::lowered::pass::AllocateBuffers>(m_buffer_scratchpad, m_is_buffer_optimized);
    pipeline.run(m_linear_ir);
}

void BufferAllocationTest::Validate() {
    std::set<size_t> gprs;
    for (const auto& expr : m_linear_ir) {
        if (const auto buffer = ov::as_type_ptr<ov::snippets::op::Buffer>(expr->get_node())) {
            gprs.insert(buffer->get_id());
        }
    }
    EXPECT_EQ(gprs.size(), m_expected_count);
    EXPECT_EQ(m_buffer_scratchpad, m_expected_size);
}

std::shared_ptr<ov::Model> EltwiseBufferAllocationTest::GetModel() const {
    const auto subtensor_eltwise = std::vector<size_t>{1, m_vector_size};
    const auto subtensor_buffer = std::vector<size_t>{ov::snippets::lowered::PortDescriptor::ServiceDimensions::FULL_DIM,
                                                      ov::snippets::lowered::PortDescriptor::ServiceDimensions::FULL_DIM};

    const auto parameter0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape({1, 3, 100, 100}));
    const auto parameter1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape({1, 3, 100, 100}));
    const auto add = std::make_shared<ov::op::v1::Add>(parameter0, parameter1);
    const auto buffer0 = std::make_shared<ov::snippets::op::IntermediateMemoryBuffer>(add, static_cast<int32_t>(subtensor_buffer.size()));
    const auto relu = std::make_shared<ov::op::v0::Relu>(buffer0);
    const auto buffer1 = std::make_shared<ov::snippets::op::IntermediateMemoryBuffer>(relu, static_cast<int32_t>(subtensor_buffer.size()));
    const auto exp = std::make_shared<ov::op::v0::Exp>(buffer1);
    const auto body = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(exp), ov::ParameterVector{parameter0, parameter1});

    MarkOp(add, subtensor_eltwise);
    MarkOp(relu, subtensor_eltwise);
    MarkOp(exp, subtensor_eltwise);
    MarkOp(buffer0, subtensor_buffer);
    MarkOp(buffer1, subtensor_buffer);

    return body;
}

void MHABufferAllocationTest::MarkBrgemm(const std::shared_ptr<ov::snippets::op::Brgemm>& node, const std::vector<size_t>& subtensor) {
    const auto subtensor_full = std::vector<size_t>{ov::snippets::lowered::PortDescriptor::ServiceDimensions::FULL_DIM,
                                                    ov::snippets::lowered::PortDescriptor::ServiceDimensions::FULL_DIM};
    ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor_ptr(
        node->input(0), std::make_shared<ov::snippets::lowered::PortDescriptor>(node->input(0), subtensor));
    ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor_ptr(
        node->input(1), std::make_shared<ov::snippets::lowered::PortDescriptor>(node->input(1), subtensor_full));
    ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor_ptr(
        node->output(0), std::make_shared<ov::snippets::lowered::PortDescriptor>(node->output(0), subtensor));
}

std::shared_ptr<ov::Model> MHABufferAllocationTest::GetModel() const {
    const auto subtensor_scalar = std::vector<size_t>{1};
    const auto subtensor_eltwise = std::vector<size_t>{1, m_vector_size};
    const auto subtensor_brgemm = std::vector<size_t>{32, ov::snippets::lowered::PortDescriptor::ServiceDimensions::FULL_DIM};
    const auto subtensor_power = std::vector<size_t>{1, ov::snippets::lowered::PortDescriptor::ServiceDimensions::FULL_DIM};

    const auto parameter0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape({1, 12, 128, 64}));
    const auto parameter1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape({1, 128, 12, 64}));
    const auto parameter2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape({1, 12, 128, 64}));

    const auto load_reshape = std::make_shared<ov::snippets::op::LoadReshape>(parameter1, 1, 0, std::vector<size_t>{0, 2, 3, 1});
    const auto store = std::make_shared<ov::snippets::op::Store>(load_reshape);
    const auto relu0 = std::make_shared<ov::op::v0::Relu>(store);
    const auto matmul0 = std::make_shared<ov::snippets::op::Brgemm>(parameter0, relu0);
    const auto relu1 = std::make_shared<ov::op::v0::Relu>(matmul0);

    // Decomposed Softmax
    const auto reduce_max = std::make_shared<ov::snippets::op::ReduceMax>(relu1, 3);
    ov::snippets::op::ReduceBase::compute_and_set_reduce_subtensors(reduce_max);
    const auto subtract = std::make_shared<ov::op::v1::Subtract>(relu1, reduce_max);
    const auto exp = std::make_shared<ov::op::v0::Exp>(subtract);

    const auto reduce_sum = std::make_shared<ov::snippets::op::ReduceSum>(exp, 3);
    ov::snippets::op::ReduceBase::compute_and_set_reduce_subtensors(reduce_sum);
    const auto power = std::make_shared<ov::snippets::op::PowerStatic>(reduce_sum, -1.f);
    const auto multiply = std::make_shared<ov::op::v1::Multiply>(exp, power);

    const auto matmul1 = std::make_shared<ov::snippets::op::Brgemm>(multiply, parameter2);
    const auto relu2 = std::make_shared<ov::op::v0::Relu>(matmul1);

    const auto body = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(relu2), ov::ParameterVector{parameter0, parameter1, parameter2});

    MarkOp(load_reshape, subtensor_scalar);
    MarkOp(store, subtensor_scalar);
    MarkOp(power, subtensor_power);

    MarkBrgemm(matmul0, subtensor_brgemm);
    MarkBrgemm(matmul1, subtensor_brgemm);

    return body;
}

TEST_P(EltwiseBufferAllocationTest, BufferAllocation) {
    Validate();
}
TEST_P(MHABufferAllocationTest, BufferAllocation) {
    Validate();
}

namespace BufferAllocationTest_Instances {

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_EltwiseNotOptimized, EltwiseBufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(false),
                                 ::testing::Values(false),  // in this test it doesn't make sense
                                 ::testing::Values(80000), // Each Buffer has own allocated memory
                                 ::testing::Values(2)),  // Each Buffer has unique ID
                         BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_EltwiseOptimized, EltwiseBufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(true),
                                 ::testing::Values(false),  // in this test it doesn't make sense
                                 ::testing::Values(40000),  // Two Buffer reuse memory
                                 ::testing::Values(1)),  // Two Buffers reuse IDs
                         BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHANotOptimizedWSplit, MHABufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(false),
                                 ::testing::Values(true),
                                 ::testing::Values(139264), // Each Buffer has own allocated memory
                                 ::testing::Values(7)),  // Each Buffer has unique ID
                         BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHAOptimizedWSplit, MHABufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(true),
                                 ::testing::Values(true),
                                 ::testing::Values(57344), // (Buffer before brgemm) + (between brgemms) + (after brgemm)
                                 ::testing::Values(2)), // (Buffer before brgemm0 and after brgemm1) + (between brgemms)
                         BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHANotOptimizedWOSplit, MHABufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(false),
                                 ::testing::Values(false),
                                 ::testing::Values(360448), // Each Buffer has own allocated memory
                                 ::testing::Values(7)),  // Each Buffer has unique ID
                         BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_MHAOptimizedWOSplit, MHABufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(true),
                                 ::testing::Values(false),
                                 ::testing::Values(98304), // (between brgemms) + (Buffer before brgemm0 and after brgemm1)
                                 ::testing::Values(2)), // (Buffer before brgemm0 and after brgemm1) + (between brgemms)
                         BufferAllocationTest::getTestCaseName);

}  // namespace BufferAllocationTest_Instances
}  // namespace snippets
}  // namespace test
}  // namespace ov

