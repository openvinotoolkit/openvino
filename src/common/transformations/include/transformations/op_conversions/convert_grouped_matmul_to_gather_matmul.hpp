// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/// @brief Converts the public op v17::GroupedMatMul into the internal op
/// ov::op::internal::GatherMatmul, which is the form consumed by the CPU/GPU
/// `GatherMatmul` nodes.
///
/// Supported cases (matching the GroupedMatMul-17 spec):
///  * Case (3D x 3D, no offsets):
///        A:[G,M,K], B:[G,K,N]            ->  out:[G,M,N]
///    Mapped as:
///        A'      = A                                 // [G, M, K]
///        B'      = Transpose(B, [0,2,1])             // [G, N, K]  (transp_b=true)
///        indices = Broadcast(Range(0,G), [M, G])     // indices[m,g] = g
///        out     = GatherMatmul(A', B', indices)     // [G, M, N]
///
///  * Case (2D x 3D with offsets):
///        A:[T,K], B:[G,K,N], offs:[G]    ->  out:[T,N]
///    Mapped as:
///        A'      = Unsqueeze(A, 0)                                  // [1, T, K]
///        B'      = Transpose(B, [0,2,1])                            // [G, N, K]
///        idx1d   = SearchSorted(offsets, Range(0,T), right=true)    // [T]
///        indices = Unsqueeze(idx1d, -1)                             // [T, 1]
///        gm      = GatherMatmul(A', B', indices)                    // [1, T, N]
///        out     = Squeeze(gm, 0)                                   // [T, N]
///
///  * Case 3 (2D x 2D weight gradient): NOT supported — left untouched.
///
/// The pass requires the weights tensor (input B) to be a Constant (or reachable
/// through a constant-foldable chain). Otherwise the public op is left in place
/// so the reference GroupedMatMul::evaluate is used.
class TRANSFORMATIONS_API ConvertGroupedMatMulToGatherMatmul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGroupedMatMulToGatherMatmul");
    ConvertGroupedMatMulToGatherMatmul();
};

}  // namespace ov::pass
