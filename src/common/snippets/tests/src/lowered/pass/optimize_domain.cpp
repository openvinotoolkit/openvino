// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "common_test_utils/common_utils.hpp"
#include "snippets/lowered/pass/optimize_domain.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "lowered/pass/optimize_domain.hpp"
#include "subgraph_simple.hpp"
#include "lowering_utils.hpp"

namespace ov {
namespace test {
namespace snippets {
OptimizeDomainParams::OptimizeDomainParams(size_t min_jit_work_amount,
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

std::string OptimizeDomainTest::getTestCaseName(testing::TestParamInfo<OptimizeDomainParams> obj) {
    OptimizeDomainParams domain_opt_params = obj.param;
    std::ostringstream result;
    result << "MinJitWork=" << domain_opt_params.min_jit_work_amount << "_";
    result << "MinParWork=" << domain_opt_params.min_parallel_work_amount << "_";
    for (size_t i = 0; i < domain_opt_params.input_shapes.size(); i++)
        result << "IS[" << i << "]=" << ov::test::utils::partialShape2str({domain_opt_params.input_shapes[i]}) << "_";
    result << "ExpMS=" << ov::test::utils::vec2str(domain_opt_params.exp_master_shape) << "_";
    result << "ExpLD=" << domain_opt_params.exp_loop_depth << "_";
    return result.str();
}

void OptimizeDomainTest::SetUp() {
    m_domain_opt_params = this->GetParam();

    ov::snippets::lowered::Config lir_config;
    lir_config.m_manual_build_support = true;
    lir_config.m_enable_domain_optimization = true;
    lir_config.m_min_parallel_work_amount = m_domain_opt_params.min_parallel_work_amount;
    lir_config.m_min_kernel_work_amount = m_domain_opt_params.min_jit_work_amount;
    m_linear_ir = std::make_shared<ov::snippets::lowered::LinearIR>(lir_config, std::make_shared<ov::snippets::IShapeInferSnippetsFactory>());

    const auto precision = ov::element::f32;
    OPENVINO_ASSERT(m_domain_opt_params.input_shapes.size() == 2);
    auto param1 = m_linear_ir->push_node<ov::op::v0::Parameter>(precision, m_domain_opt_params.input_shapes[0]);
    auto param2 = m_linear_ir->push_node<ov::op::v0::Parameter>(precision, m_domain_opt_params.input_shapes[1]);
    auto add = m_linear_ir->push_node<ov::op::v1::Add>(param1.second, param2.second);
    auto result = m_linear_ir->push_node<ov::op::v0::Result>(add.second);
}

TEST_P(OptimizeDomainTest, DomainOptimization) {
    size_t loop_depth = 1;
    ov::snippets::lowered::pass::OptimizeDomain(loop_depth).run(*m_linear_ir);
    const auto& master_shape = m_linear_ir->get_master_shape();
    EXPECT_EQ(loop_depth, m_domain_opt_params.exp_loop_depth) << "Inconsistent loop depth detected";
    EXPECT_THAT(master_shape, testing::ContainerEq(m_domain_opt_params.exp_master_shape)) << "Inconsistent master_shape detected";
}

namespace OptimizeDomainTestsInstantiation {

std::vector<OptimizeDomainParams> dopt_params = {
        // No broadcasting => dimensions collapsed
        {256, 4, {{14, 15, 1, 17}, {14, 15, 1, 17}}, {1, 1, 14, 255}, 1},
        {256, 4, {{14, 15, 16, 1}, {14, 15, 16, 1}}, {1, 1, 14, 240}, 1},
        // Same dimensions, but larger num threads => collapsing omitted
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

        // Broadcasting breaks dimension collapsing after first collapsion => loop depth incremented
        {256, 4, {{14, 15, 16, 16}, {16, 16}}, {1, 14, 15, 256}, 2},

        // Collapse even if not enough work to cover min_jit_work_amount
        {256, 18, {{4, 5, 6, 7}, {4, 5, 6, 7}}, {1, 4, 5, 42}, 1},
        // Same dims, but higher parallel work amount => do not collapse to load all the threads
        {256, 32, {{4, 5, 6, 7}, {4, 5, 6, 7}}, {4, 5, 6, 7}, 1},

        // 2D and 1D shapes are too small, so no collapsing should be done in such cases
        {256, 2, {{256, 256, 3}, {3}}, {256, 256, 3}, 2},
        {256, 32, {{4, 5}, {4, 5}}, {4, 5}, 1},
        {256, 32, {{5}, {5}}, {5}, 1},

        // min_parallel_work_amount = 1 is a special case that would cause all dimensions to collapse (up to min_jit_work_amount of course)
        {256, 1, {{4, 1, 6, 7}, {4, 1, 6, 7}}, {1, 1, 1, 168}, 1},
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_DomainOptimization, OptimizeDomainTest,
                         ::testing::ValuesIn(dopt_params),
                         OptimizeDomainTest::getTestCaseName);

} // namespace OptimizeDomainTestsInstantiation
}  // namespace snippets
}  // namespace test
}  // namespace ov