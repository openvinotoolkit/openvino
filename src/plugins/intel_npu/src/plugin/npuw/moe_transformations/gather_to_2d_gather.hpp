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
 *   1. reshaped_indices = Reshape(indices[I]) -> [I, 1]
 *   2. experts_start = Multiply(reshaped_indices, M) -> [I, 1]
 *   3. new_indices = Add(experts_start, range_M) -> [I, M]
 *   4. flat_indices = Reshape(new_indices) -> [I*M]
 *   5. flat_weights = Reshape(weights) -> [N*M, K]
 *   6. gathered = Gather(flat_weights, flat_indices) -> [I*M, K]
 *   7. output = Reshape(gathered) -> [I, M, K]
 *
 * This transformation enables better hardware support for Gather operations
 * by converting multi-dimensional gather into flattened 2D gather.
 */
class GatherTo2DGather : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("GatherTo2DGather", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace npuw
}  // namespace ov
