// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::npuw {

/**
 * @brief Folds a single-element, all-Constant dequantization chain into one Constant.
 *
 * Per-tensor quantized scalars (e.g. ONNX DequantizeLinear of a norm scale) keep several tiny
 * Constants and Convert/Subtract/Multiply nodes for a value known at compile time. Folding them
 * leaves only real compressed weights as DQ chains for the following NPUW weight passes.
 *
 * Before:                                              After:
 *
 *   Const(w)   Const(zp)                                Const((w - zp) * scale)
 *      │           │                                             │
 *   Convert   [Convert]                                          │
 *      └─────┬─────┘                                             │
 *       [Subtract]   Const(scale)                                │
 *            │          │                                        │
 *            │      [Convert]                                    │
 *            └────┬─────┘                                        │
 *             Multiply                                           │
 *                 │                                              │
 *
 * [ ] marks optional nodes; Subtract and Multiply inputs may be in either order. Applied only when the
 * Multiply output has a static shape with exactly one element, so compressed weights stay
 * untouched.
 */
class FoldScalarDQ : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ov::npuw::FoldScalarDQ");
    FoldScalarDQ();
};

}  // namespace ov::npuw
