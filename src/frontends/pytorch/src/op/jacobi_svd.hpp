// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

struct JacobiLoopResult {
    Output<Node> a;  // rotated working matrix (B, rows, cols)
    Output<Node> v;  // accumulated right factor (B, cols, cols); empty when accumulate_v is false
};

// Runs the tiled one-sided Jacobi column-rotation Loop over a rank-3 (B, rows, cols) matrix. It
// orthogonalizes the columns by a sequence of Givens rotations applied on the right: the schedule is
// the upper-triangle pair list (p < q, length cols*(cols-1)/2) tiled `sweeps` times, driven by one
// v5::Loop that rotates columns p, q per iteration (a CPU-plugin loop body cannot have a dynamic
// rank, hence the fixed rank-3 shape). When accumulate_v, the same rotations accumulate into an
// identity-seeded V (cols x cols). Shared by aten::svd (jacobi_svd) and the values-only matrix
// svdvals used by the spectral/nuclear norms.
JacobiLoopResult run_jacobi_column_loop(const NodeContext& context,
                                        const Output<Node>& A_flat,
                                        const Output<Node>& n,
                                        int sweeps,
                                        element::Type et,
                                        bool accumulate_v);

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
