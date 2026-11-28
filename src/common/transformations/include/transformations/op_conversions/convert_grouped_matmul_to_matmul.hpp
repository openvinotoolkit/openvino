// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/// @brief Decomposes the 3Dx3D v17::GroupedMatMul (no offsets) into a single batched MatMul.
///
///     A:[G,M,K], B:[G,N,K]  ->  out:[G,M,N]
///     out = MatMul(A, B, transpose_a=false, transpose_b=true)    // B is stored transposed
class TRANSFORMATIONS_API ConvertGroupedMatMul3Dx3DToMatMul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGroupedMatMul3Dx3DToMatMul");
    ConvertGroupedMatMul3Dx3DToMatMul();
};

/// @brief Decomposes the 2Dx3D v17::GroupedMatMul (with offsets) into per-group MatMuls.
///
///     A:[T,K], B:[G,N,K], offs:[G]  ->  out:[T,N]
///     `offsets` holds cumulative end offsets, so the per-group row counts are
///     `offsets - [0, offsets[:-1]]`. A is partitioned by those lengths and each
///     partition is multiplied by its own weight matrix:
///         A_0..A_{G-1} = VariadicSplit(A, axis=0, split_lengths)   // [Mg, K]
///         B_0..B_{G-1} = Squeeze(Split(B, axis=0, G), axis=0)      // [N, K]
///         out_g        = MatMul(A_g, B_g, transpose_a=false, transpose_b=true)
///         out          = Concat(out_0, ..., out_{G-1}, axis=0)     // [T, N]
///     Requires the group dimension G (mat_b.shape[0]) to be static so that a
///     fixed number of MatMuls can be generated.
class TRANSFORMATIONS_API ConvertGroupedMatMul2Dx3DToMatMul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGroupedMatMul2Dx3DToMatMul");
    ConvertGroupedMatMul2Dx3DToMatMul();
};

/// @brief Decomposes the public op v17::GroupedMatMul into a regular v0::MatMul operations graph,
/// covering both the 3Dx3D and 2Dx3D (with offsets) cases.
class TRANSFORMATIONS_API ConvertGroupedMatMulToMatMul : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ConvertGroupedMatMulToMatMul");
    ConvertGroupedMatMulToMatMul() {
        add_matcher<ConvertGroupedMatMul3Dx3DToMatMul>();
        add_matcher<ConvertGroupedMatMul2Dx3DToMatMul>();
    }
};

}  // namespace ov::pass
