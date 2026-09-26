// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace npuw {
namespace pass {

/**
 * @brief Transform 3D Gather to 2D Gather sequence for hardware optimization
 *
 * This pass transforms:
 *   Gather(weights[N, M, K], indices[I]) -> output[I, M, K]
 *
 * Into equivalent sequence:
 *   0. indices_i32 = Convert(indices[I]) -> [I] (i32) -- only inserted when the source
 *      indices are not already i32 (e.g. i64)
 *   1. reshaped_indices = Reshape(indices_i32) -> [I, 1]
 *   2. experts_start = Multiply(reshaped_indices, M) -> [I, 1]
 *   3. range_m = Constant([0, 1, ..., M-1]) -> [1, M]
 *   4. new_indices = Add(experts_start, range_m) -> [I, M]
 *   5. flat_indices = Reshape(new_indices) -> [I*M]
 *   6. flat_weights = Reshape(weights) -> [N*M, K]
 *   7. gathered_flat = Gather(flat_weights, flat_indices, axis=0) -> [I*M, K]
 *   8. output = Reshape(gathered_flat) -> [I, M, K]
 *
 * This transformation enables better hardware support for Gather operations
 * by converting multi-dimensional gather into flattened 2D gather.
 *
 * Gathers that read the exact same indices tensor share steps 0-4 -- built once per group
 * instead of once per Gather: members with equal M reuse the identical [I, M] tensor directly,
 * members with different M each get their own slice out of a single combined Multiply/Add via
 * VariadicSplit. A Gather with no siblings sharing its indices degenerates to exactly the
 * single-Gather sequence above.
 */
class GatherTo2DGather : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("GatherTo2DGather", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace npuw
}  // namespace ov
