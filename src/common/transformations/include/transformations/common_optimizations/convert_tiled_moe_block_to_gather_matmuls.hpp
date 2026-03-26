// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

// BGM-producing passes (IR → GatherMatmul)
class TRANSFORMATIONS_API ConvertTiledMoeBlockTo2GatherMatmuls;
class TRANSFORMATIONS_API ConvertTiledMoeBlockTo3GatherMatmuls;

class TRANSFORMATIONS_API ConvertTiledMoeBlockToGatherMatmuls;

}  // namespace pass
}  // namespace ov

// ============================================================================
// ConvertTiledMoeBlockTo2GatherMatmuls
// ============================================================================
//
// Matches tiled MoE blocks where gate+up are fused into a single MatMul,
// then sliced apart (common in models with combined gate/up projection).
//
// BEFORE:
//
//   experts_input
//        |
//     Reshape
//        |
//      Tile  (replicate across experts)
//        |
//     Reshape  [batch*seq*n_experts, hidden]
//        |
//   gate_up_MatMul (transpose_b)        chosen_experts
//        |                                     |
//   gate_up_Add  (bias, 2 consumers)    ScatterElementsUpdate
//      /       \                               |
//   Slice1    Slice2                       Transpose
//     |         |                              |
//   Clamp    Minimum                       Reshape
//     |         |                              |
//    Add      Swish                       (Unsqueeze)
//      \       /                               |
//      Multiply                                |
//         |                                    |
//   down_MatMul (transpose_b)                  |
//         |                                    |
//   down_Add (bias)                            |
//         |                                    |
//      Reshape                                 |
//         |                                    |
//      Multiply  <-----------------------------+
//         |
//      ReduceSum (experts dim)
//
//
// AFTER:
//
//   experts_input
//        |
//     Unsqueeze(0)     active_indices    chosen_experts
//        |                   |                 |
//   GatherMatmul_1 (gate+up weights, bias) Transpose
//      /       \                               |
//   Slice1    Slice2                        Unsqueeze
//     |         |                              |
//   Clamp    Minimum                           |
//     |         |                              |
//    Add      Swish                            |
//      \       /                               |
//      Multiply                                |
//        |          active_indices             |
//        |               |                     |
//   GatherMatmul_2 (down weights, bias)        |
//        |                                     |
//      Multiply  <-----------------------------+
//        |
//      ReduceSum
//        |
//      Reshape  (remove experts dim)
//
// ============================================================================
class ov::pass::ConvertTiledMoeBlockTo2GatherMatmuls : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertTiledMoeBlockTo2GatherMatmuls");
    ConvertTiledMoeBlockTo2GatherMatmuls();
};

// ============================================================================
// ConvertTiledMoeBlockTo3GatherMatmuls
// ============================================================================
//
// Matches tiled MoE blocks with separate gate, up, and down MatMuls
// (SwiGLU architecture — no biases, no slicing).
//
// BEFORE:
//
//   experts_input
//        |
//     Reshape
//        |
//      Tile  (replicate across experts)
//        |
//     Reshape  [batch*seq*n_experts, hidden]
//      /               |
//  gate_MatMul      up_MatMul           chosen_experts
//     |                |                      |
//   Swish              |              ScatterElementsUpdate
//      \              /                       |
//       Multiply  (SwiGLU)                Transpose
//           |                                 |
//       down_MatMul                       Reshape
//           |                                 |
//        Reshape                          (Unsqueeze)
//           |                                 |
//        Multiply  <--- router_weights -------+
//           |
//        ReduceSum (experts dim)
//
//
// AFTER:
//
//   experts_input
//        |
//     Unsqueeze(0)          active_indices   chosen_experts
//      /        \                 |               |
//  GatherMatmul_1(gate)   GatherMatmul_2(up)  Transpose
//     |                        |                  |
//   Swish                      |               Unsqueeze
//      \                      /                   |
//       Multiply  (SwiGLU)                        |
//         |            active_indices             |
//         |                 |                     |
//    GatherMatmul_3 (down weights)                |
//         |                                       |
//      Multiply  <--------------------------------+
//         |
//      ReduceSum
//         |
//      Reshape  (remove experts dim)
//
// ============================================================================
class ov::pass::ConvertTiledMoeBlockTo3GatherMatmuls : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertTiledMoeBlockTo3GatherMatmuls");
    ConvertTiledMoeBlockTo3GatherMatmuls();
};

// CPU uses BGM-producing passes only (stops at BGMs)
class ov::pass::ConvertTiledMoeBlockToGatherMatmuls : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ConvertTiledMoeBlockToGatherMatmuls");
    ConvertTiledMoeBlockToGatherMatmuls() {
        add_matcher<ov::pass::ConvertTiledMoeBlockTo2GatherMatmuls>();
        add_matcher<ov::pass::ConvertTiledMoeBlockTo3GatherMatmuls>();
    }
};
