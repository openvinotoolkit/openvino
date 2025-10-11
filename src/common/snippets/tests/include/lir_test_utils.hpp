// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <vector>

#include "common_test_utils/test_common.hpp"
#include "lir_comparator.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/pass/pass.hpp"

namespace ov {
namespace test {
namespace snippets {
class LoweredPassTestsF : public ov::test::TestsCommon {
public:
    LoweredPassTestsF();

    void SetUp() override {}

    void TearDown() override;

    /**
     * @brief Helper method to reorder loop IDs and assign them to expressions in reference LIR
     * @param expr_to_loop_ids Map from expressions to their loop IDs in original notation
     * @param loop_ids_mapper Map for reordering loop IDs to match the actual LIR after transformations
     *
     * Note: Since it's impossible to set the desired loop ID during loop info creation,
     * we have to reorder the loop IDs in reference LIR to make them match the actual LIR.
     * This helper is needed only for linear_ir_ref (expected result) because the actual LIR
     * (linear_ir) gets its loop IDs assigned/fused/split automatically, while
     * the reference LIR needs manual configuration to match the expected transformation result.
     *
     * This helper encapsulates the logic of:
     * 1. Reordering loop identifiers in the loop manager using the provided mapper
     * 2. Setting the reordered loop IDs to the corresponding expressions
     */
    void assign_loop_ids(const std::map<ov::snippets::lowered::ExpressionPtr, std::vector<size_t>>& expr_to_loop_ids,
                         const std::map<size_t, size_t>& loop_ids_mapper);

    std::shared_ptr<ov::snippets::lowered::LinearIR> linear_ir, linear_ir_ref;
    ov::snippets::lowered::pass::PassPipeline pipeline;
    LIRComparator comparator;
};

/**
 * @brief Returns default subtensor with passed rank filled with FULL_DIM values.
 * @param rank rank of subtensor
 * @return default subtensor
 */
ov::snippets::VectorDims get_default_subtensor(size_t rank = 2);

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

using IOLoopPortDescs = std::pair<std::vector<ov::snippets::lowered::UnifiedLoopInfo::LoopPortDesc>,
                                  std::vector<ov::snippets::lowered::UnifiedLoopInfo::LoopPortDesc>>;
/**
 * @brief Creates an InnerSplittedUnifiedLoopInfo which represents an inner loop that appears
 * after SplitLoops optimizations.
 *
 * @param work_amount work amount of original loop before splitting
 * @param increment increment for each iteration
 * @param entries Vector of LoopPort objects representing loop entry points (input ports)
 * @param exits Vector of LoopPort objects representing loop exit points (output ports)
 * @param outer_split_loop_info Pointer to the outer split loop info that will contain this inner loop
 * @param io_descs Optional parameter containing input and output port descriptors
 * @return Shared pointer to the created InnerSplittedUnifiedLoopInfo
 */
ov::snippets::lowered::InnerSplittedUnifiedLoopInfoPtr make_inner_split_loop_info(
    size_t work_amount,
    size_t increment,
    const std::vector<ov::snippets::lowered::LoopPort>& entries,
    const std::vector<ov::snippets::lowered::LoopPort>& exits,
    const ov::snippets::lowered::UnifiedLoopInfoPtr& outer_split_loop_info,
    const std::optional<IOLoopPortDescs>& io_descs = std::nullopt);

}  // namespace snippets
}  // namespace test
}  // namespace ov
