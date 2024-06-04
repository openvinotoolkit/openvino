// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common_test_utils/ov_test_utils.hpp>

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/shape_types.hpp"

namespace ov {
namespace test {
namespace snippets {

struct OptimizeDomainParams {
    OptimizeDomainParams() = default;
    OptimizeDomainParams(size_t, size_t, std::vector<ov::PartialShape>, ov::snippets::VectorDims, size_t);
    size_t min_jit_work_amount = 0;
    size_t min_parallel_work_amount = 0;
    std::vector<ov::PartialShape> input_shapes;
    ov::snippets::VectorDims exp_master_shape;        // Expected master_shape
    size_t exp_loop_depth = 0;                        // Expected loop depth (aka tile rank)
};

class OptimizeDomainTest : public testing::TestWithParam<OptimizeDomainParams> {
public:
    using VectorDims = ov::snippets::VectorDims;
    static std::string getTestCaseName(testing::TestParamInfo<OptimizeDomainParams> obj);
protected:
    void SetUp() override;
    ov::snippets::lowered::LinearIRPtr m_linear_ir;
    OptimizeDomainParams m_domain_opt_params;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
