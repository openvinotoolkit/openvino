// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lowered/pass/buffer_allocation.hpp"

#include "openvino/opsets/opset.hpp"
#include "snippets/snippets_isa.hpp"
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

#include "common_test_utils/common_utils.hpp"


namespace ov {
namespace test {
namespace snippets {

std::string BufferAllocationTest::getTestCaseName(testing::TestParamInfo<ov::test::snippets::BufferAllocationParams> obj) {
    bool is_optimized, with_split_loops;
    size_t expected_size, expected_reg_group_count, expected_cluster_count;

    std::tie(is_optimized, with_split_loops, expected_size, expected_reg_group_count, expected_cluster_count) = obj.param;

    std::ostringstream result;
    result << "Opt=" << ov::test::utils::bool2str(is_optimized) << "_";
    result << "Split=" << ov::test::utils::bool2str(with_split_loops) << "_";
    result << "ExpBufferSize=" << expected_size << "_";
    result << "ExpBufferRegGroupCount=" << expected_reg_group_count << "_";
    result << "ExpBufferClustersCount=" << expected_reg_group_count << "_";
    return result.str();
}

void BufferAllocationTest::SetUp() {
    std::tie(m_is_buffer_optimized, m_with_split_loops, m_expected_size, m_expected_reg_group_count, m_expected_cluster_count) = this->GetParam();

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
    pipeline.register_pass<ov::snippets::lowered::pass::InsertBuffers>();
    pipeline.register_pass<ov::snippets::lowered::pass::InsertLoadStore>(m_vector_size);
    pipeline.register_pass<ov::snippets::lowered::pass::InitLoops>();
    pipeline.register_pass<ov::snippets::lowered::pass::InsertLoops>();
    pipeline.register_pass<ov::snippets::lowered::pass::AllocateBuffers>(m_is_buffer_optimized);
    pipeline.run(m_linear_ir);
}

void BufferAllocationTest::Validate() {
    std::set<size_t> reg_groups, clusters;
    for (const auto& buffer_expr : m_linear_ir.get_buffers()) {
        reg_groups.insert(buffer_expr->get_reg_group());
        clusters.insert(buffer_expr->get_cluster_id());
    }
    EXPECT_EQ(reg_groups.size(), m_expected_reg_group_count);
    EXPECT_EQ(clusters.size(), m_expected_cluster_count);
    EXPECT_EQ(m_linear_ir.get_static_buffer_scratchpad_size(), m_expected_size);
}

std::shared_ptr<ov::Model> EltwiseBufferAllocationTest::GetModel() const {
    const auto subtensor_eltwise = std::vector<size_t>{1, m_vector_size};
    const auto subtensor_buffer = std::vector<size_t>(2, ov::snippets::utils::get_full_dim_value());

    const auto parameter0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape({1, 3, 100, 100}));
    const auto parameter1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape({1, 3, 100, 100}));
    const auto add = std::make_shared<ov::op::v1::Add>(parameter0, parameter1);
    const auto buffer0 = std::make_shared<ov::snippets::op::Buffer>(add);
    const auto relu = std::make_shared<ov::op::v0::Relu>(buffer0);
    const auto buffer1 = std::make_shared<ov::snippets::op::Buffer>(relu);
    const auto exp = std::make_shared<ov::op::v0::Exp>(buffer1);
    const auto body = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(exp), ov::ParameterVector{parameter0, parameter1});

    MarkOp(add, subtensor_eltwise);
    MarkOp(relu, subtensor_eltwise);
    MarkOp(exp, subtensor_eltwise);
    MarkOp(buffer0, subtensor_buffer);
    MarkOp(buffer1, subtensor_buffer);

    return body;
}

TEST_P(EltwiseBufferAllocationTest, BufferAllocation) {
    Validate();
}

namespace BufferAllocationTest_Instances {

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_EltwiseNotOptimized, EltwiseBufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(false),
                                 ::testing::Values(false),  // in this test it doesn't make sense
                                 ::testing::Values(80000),  // Each Buffer has own allocated memory
                                 ::testing::Values(2),      // Each Buffer has unique reg group
                                 ::testing::Values(2)),     // Each Buffer has unique cluster ID
                         BufferAllocationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BufferAllocation_EltwiseOptimized, EltwiseBufferAllocationTest,
                         ::testing::Combine(
                                 ::testing::Values(true),
                                 ::testing::Values(false),  // in this test it doesn't make sense
                                 ::testing::Values(40000),  // Two Buffer reuse memory
                                 ::testing::Values(1),      // Two Buffers reuse IDs
                                 ::testing::Values(1)),     // Two Buffers are from the same luster
                         BufferAllocationTest::getTestCaseName);

}  // namespace BufferAllocationTest_Instances
}  // namespace snippets
}  // namespace test
}  // namespace ov

