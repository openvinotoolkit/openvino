// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"
#include "snippets/shape_types.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface OptimizeDomain
 * @brief Collapse input/output dimensions to balance parallel/per-thread load. The pass consists of two steps:
 *        The pass collapses two last dimensions while none of them is broadcasted and the resulting dim size
 *        1. Dimension collapsing: If none of the last two dimensions are broadcasted, the last dimension's size
 *           is less than min_kernel_work_amount and the remaining dimensions provide work amount larger than
 *           min_parallel_work_amount (min_kernel_work_amount and min_parallel_work_amount specified in LireanIR config),
 *           then these two dimensions are collapsed into one and the collapsing attempt is repeated.
 *        2. Tile rank increment: Tile rank is the rank of a tensor that processed during one call. If all except
 *           for the last two dimensions provide work_amount larger than min_parallel_work_amount, then tile_rank
 *           is incremented. This effectively increases kernel work_amount.
 *        Examples of graphs before and after this transformations are depicted below.
 * @param tile_rank (taken by reference) rank of a tensor that processed during one call. Incremented if dimensions are collapsed.
 * @ingroup snippets
 */
// Example:
// min_jit_work_amount = 256
// min_parallel_work_amount = 4
//
//          Before OptimizeDomain         |      After OptimizeDomain
// -------------------------------------------------------------------
// tile_rank = 1                          |   tile_rank = 2
//                                        |
//       in1            in2               |         in1            in2
// [14, 15, 16, 17]  [14, 15, 16, 17]     |   [1, 14, 15, 272]  [1, 14, 15, 272]
//             \      /                   |               \      /
//                Add                     |                  Add
//          [14, 15, 16, 17]              |            [1, 14, 15, 272]
//                 |                      |                   |
//               Result                   |                 Result
//           [14, 15, 16, 17]             |             [1, 14, 15, 272]

class OptimizeDomain : public snippets::lowered::pass::Pass {
public:
    OPENVINO_RTTI("OptimizeDomain", "", Pass)
    explicit OptimizeDomain(size_t& tile_rank);
    bool run(LinearIR& linear_ir) override;

private:
    size_t& m_tile_rank;
    static size_t optimize(std::vector<VectorDims>& input_shapes,
                           VectorDims& master_shape,
                           size_t total_work_amount,
                           size_t min_parallel_work_amount,
                           size_t min_jit_work_amount);
    inline static bool can_increase_jit_work_amount(const VectorDims& master_shape,
                                                    size_t min_parallel_work_amount,
                                                    size_t total_work_amount);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov