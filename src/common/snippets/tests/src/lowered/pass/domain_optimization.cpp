// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "common_test_utils/common_utils.hpp"
#include "snippets/lowered/pass/domain_optimization.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "lowered/pass/domain_optimization.hpp"
#include "subgraph_simple.hpp"
#include "lowering_utils.hpp"

namespace ov {
namespace test {
namespace snippets {
DomainOptimizationParams::DomainOptimizationParams(size_t min_jit_work_amount,
                                                   size_t min_parallel_work_amount,
                                                   std::vector<ov::PartialShape> input_shapes,
                                                   ov::snippets::VectorDims exp_master_shape,
                                                   size_t exp_loop_depth) :
                                                   min_jit_work_amount(min_jit_work_amount),
                                                   min_parallel_work_amount(min_parallel_work_amount),
                                                   input_shapes(std::move(input_shapes)),
                                                   exp_master_shape(std::move(exp_master_shape)),
                                                   exp_loop_depth(exp_loop_depth) {
}

std::string DomainOptimizationTest::getTestCaseName(testing::TestParamInfo<DomainOptimizationParams> obj) {
    DomainOptimizationParams domain_opt_params = obj.param;
    std::ostringstream result;
    result << "MinJitWork=" << domain_opt_params.min_jit_work_amount << "_";
    result << "MinParWork=" << domain_opt_params.min_parallel_work_amount << "_";
    for (size_t i = 0; i < domain_opt_params.input_shapes.size(); i++)
        result << "IS[" << i << "]=" << ov::test::utils::partialShape2str({domain_opt_params.input_shapes[i]}) << "_";
    result << "ExpMS=" << ov::test::utils::vec2str(domain_opt_params.exp_master_shape) << "_";
    result << "ExpLD=" << domain_opt_params.exp_loop_depth << "_";
    return result.str();
}

void DomainOptimizationTest::SetUp() {
    m_domain_opt_params = this->GetParam();
    m_model = std::make_shared<EltwiseFunction>(m_domain_opt_params.input_shapes)->getOriginal();
}

TEST_P(DomainOptimizationTest, DomainOptimization) {
    auto subgraph = LoweringTests::getTokenizedSubgraph(m_model);
    subgraph->set_min_jit_work_amount(m_domain_opt_params.min_jit_work_amount);
    subgraph->set_min_parallel_work_amount(m_domain_opt_params.min_parallel_work_amount);
    auto linear_ir = *subgraph->convert_body_to_linear_ir();
    size_t loop_depth = 1;
    ov::snippets::lowered::pass::PassPipeline domain_optimization_pipeline;
    domain_optimization_pipeline.register_pass<ov::snippets::lowered::pass::DomainOptimization>(loop_depth);
    domain_optimization_pipeline.run(linear_ir);
    const auto& master_shape = linear_ir.get_master_shape();
    EXPECT_EQ(loop_depth, m_domain_opt_params.exp_loop_depth) << "Inconsistent loop depth detected";
    EXPECT_THAT(master_shape, testing::ContainerEq(m_domain_opt_params.exp_master_shape)) << "Inconsistent master_shape detected";
}

namespace DomainOptimizationTestsInstantiation {

std::vector<DomainOptimizationParams> dopt_params = {
        // todo: Discuss on review: we collapse 3 dims here and we've never done it before. concerns?
        // No broadcasting => dimensions collapsed
        {256, 4, {{14, 15, 1, 17}, {14, 15, 1, 17}}, {1, 1, 14, 255}, 1},
        {256, 4, {{14, 15, 16, 1}, {14, 15, 16, 1}}, {1, 1, 14, 240}, 1},
        // Same dimensions, but larger num threads => collapsing omitted
        // todo: Discuss on review: we eliminate 1 dims here and we've never done it before. concerns?
        {256, 18, {{14, 15, 1, 17}, {14, 15, 1, 17}}, {1, 14, 15, 17}, 1},
        {256, 18, {{14, 15, 16, 1}, {14, 15, 16, 1}}, {1, 14, 15, 16}, 1},

        // No broadcasting => collapsing and loop_depth increment
        {256, 4, {{14, 15, 16, 17}, {14, 15, 16, 17}}, {1, 14, 15, 272}, 2},
        // Same dimensions, but smaller jit work amount => collapsing omitted
        {16, 4, {{14, 15, 16, 17}, {14, 15, 16, 17}}, {14, 15, 16, 17}, 2},
        // Same dimensions, but higher parallel work amount => collapsing but no loop_depth increment
        {256, 18, {{14, 15, 16, 17}, {14, 15, 16, 17}}, {1, 14, 15, 272}, 1},

        // Broadcasting breaks dimension collapsing => loop depth incremented
        {256, 4, {{14, 15, 16, 1}, {14, 15, 1, 17}}, {14, 15, 16, 17}, 2},
        {256, 4, {{14, 15, 1, 17}, {14, 15, 16, 17}}, {14, 15, 16, 17}, 2},

        // Collapse even if not enough work to cover min_jit_work_amount
        {256, 18, {{4, 5, 6, 7}, {4, 5, 6, 7}}, {1, 4, 5, 42}, 1},
        // Same dims, but higher parallel work amount => do not collapse to load all the threads
        {256, 32, {{4, 5, 6, 7}, {4, 5, 6, 7}}, {4, 5, 6, 7}, 1},
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_DomainOptimization, DomainOptimizationTest,
                         ::testing::ValuesIn(dopt_params),
                         DomainOptimizationTest::getTestCaseName);

} // namespace DomainOptimizationTestsInstantiation
}  // namespace snippets
}  // namespace test
}  // namespace ov