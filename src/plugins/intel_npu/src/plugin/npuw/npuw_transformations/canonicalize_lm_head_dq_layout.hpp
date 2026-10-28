// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::npuw {

/**
 * @brief Brings the LM-head dequantize chain emitted by the WebNN / ORT export to the
 *        layout produced by optimum-intel.
 *
 * The exported chain wraps the u8 per-channel weight, zero point and scale in extra
 * Reshape / Transpose ops, which the tail-vocab matchers (e.g. PreserveConstDictMatMulAsymm)
 * don't recognize: the weight is then lifted to a weights-bank Parameter and the NPU compiler
 * falls back to a runtime dequantize.
 *
 * Before (OC - vocab size, IC - hidden size):
 *
 *   W:u8 [OC,1,IC]    Z:u8 [OC]                   S:f16 [OC]
 *        │               │                            │
 *     Convert    Reshape [OC,1,1] / Convert ...   Reshape [OC,1,1]
 *        │               │                            │
 *        └───Subtract────┘                            │
 *                │                                    │
 *                └────────────Multiply────────────────┘
 *                                │
 *                          Reshape [OC,IC]
 *                                │
 *                         Transpose {1,0}    Act
 *                                │            │
 *                                └───MatMul───┘
 *                                (transpose_b = false)
 *
 * After:
 *
 *   W:u8 [OC,IC]      Z:u8 [OC,1]
 *        │               │
 *     Convert         Convert
 *        │               │
 *        └───Subtract────┘     S:f16 [OC,1]
 *                │                  │
 *                └─────Multiply─────┘
 *                        │          Act
 *                        └──MatMul───┘
 *                      (transpose_b = true)
 *
 * The Reshapes are applied to the (still compressed) Constants, so the weight stays u8 and its
 * data buffer is shared, not copied. The Reshape on the zero point may be placed either before
 * or after its Convert.
 *
 * Applied only when the weight, zero point and scale are per-channel along the first (OC)
 * dimension and the chain has exactly the shape above; any other graph is left untouched.
 */
class CanonicalizeLMHeadDQLayout : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::CanonicalizeLMHeadDQLayout");
    CanonicalizeLMHeadDQLayout();
};

}  // namespace ov::npuw
