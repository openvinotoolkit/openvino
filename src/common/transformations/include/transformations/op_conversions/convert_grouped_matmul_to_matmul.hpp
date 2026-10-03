// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/// @brief Decomposes the public op v17::GroupedMatMul into regular v0::MatMul
/// operations graph.
///
/// Supported cases (matching the GroupedMatMul-17 spec):
///  * Case (3D x 3D, no offsets):
///        A:[G,M,K], B:[G,N,K]            ->  out:[G,M,N]
///    Mapped as a single batched MatMul (B is stored transposed):
///        out = MatMul(A, B, transpose_a=false, transpose_b=true)    // [G, M, N]
///
///  * Case (2D x 3D with offsets):
///        A:[T,K], B:[G,N,K], offs:[G]    ->  out:[T,N]
///    The rows of A are partitioned per expert by `offsets` (cumulative end
///    offsets) and each partition is multiplied by its own weight matrix:
///        for g in [0, G):
///            start_g = (g == 0) ? 0 : offsets[g-1]
///            end_g   = offsets[g]
///            A_g     = Slice(A, start_g, end_g, axis=0)             // [Mg, K]
///            B_g     = Gather(B, g, axis=0)                         // [N, K]
///            out_g   = MatMul(A_g, B_g, transpose_a=false, transpose_b=true)
///        out = Concat(out_0, ..., out_{G-1}, axis=0)               // [T, N]
///    This case requires the group dimension G (mat_b.shape[0]) to be static so
///    that a fixed number of MatMuls can be generated.
class TRANSFORMATIONS_API ConvertGroupedMatMulToMatMul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGroupedMatMulToMatMul");
    ConvertGroupedMatMulToMatMul();
};

}  // namespace ov::pass
