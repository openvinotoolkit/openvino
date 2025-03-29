// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/test_common.hpp"
#include "lir_comparator.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/pass.hpp"

namespace ov {
namespace test {
namespace snippets {
class LoweredPassTestsF : public ov::test::TestsCommon {
public:
    LoweredPassTestsF();

    void SetUp() override {}

    void TearDown() override;

    std::shared_ptr<ov::snippets::lowered::LinearIR> linear_ir, linear_ir_ref;
    ov::snippets::lowered::pass::PassPipeline pipeline;
    LIRComparator comparator;
};

/**
 * @brief Returns default 2D subtensor filled with FULL_DIM values.
 * @return default subtensor
 */
ov::snippets::VectorDims get_default_subtensor();

/**
 * @brief Inits input and output descriptors, and sets them to expression and its ov::Node.
 * @attention Descriptor shapes are initialized using ov::Node input/output shapes
 * @attention If optional vector of parameters (subtensors or layouts) is set, its size must be equal to n_inputs + n_outputs
 * @attention If subtensors are not set, default 2D subtensor (filled with FULL_DIM values) is created
 * @param expr expression whose descriptors should be initialized
 * @param subtensors vector of subtensors to set
 * @param layouts vector of layouts to set
 */
void init_expr_descriptors(const ov::snippets::lowered::ExpressionPtr& expr,
                           const std::vector<ov::snippets::VectorDims>& subtensors = {},
                           const std::vector<ov::snippets::VectorDims>& layouts = {});

}  // namespace snippets
}  // namespace test
}  // namespace ov
