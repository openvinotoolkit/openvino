// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

/**
 * @interface MarkCompressedWeightsCast
 * @brief Marks the trailing precision cast of a weight-decompression subgraph as decompression.
 *
 * A model emitted in a dtype other than the one its weights are dequantized in ends the
 * decompression chain with a cast. A bf16 torch model with int4 weights, for instance, arrives as
 *
 *     Constant(i4) -> Convert(f16) -> [Subtract(zp)] -> Multiply(scale) -> [Reshape/Transpose]
 *                  -> Convert(bf16) -> MatMul
 *
 * Nothing marks that final Convert as decompression: ov::pass::MarkCompressedFloatConstants only
 * marks Converts that target f32 and sit directly on a Constant, and ov::pass::MarkDequantization
 * uses the separate dequantization attribute. Consumers that gate on ov::is_decompression() --
 * notably the CPU plugin's ConvertMatMulToFC -- therefore reject such weights, and the dequant is
 * left in the graph as a runtime Convert + Multiply per layer instead of folding into
 * FullyConnectedCompressed.
 *
 * Marking here makes those consumers accept the chain. The fold itself needs nothing further:
 * ov::pass::pattern::op::CompressedWeightsBlock already ends in an optional trailing Convert, with
 * no predicate on it, so ConvertFullyConnectedToFullyConnectedCompressed absorbs the cast.
 */
class MarkCompressedWeightsCast : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::MarkCompressedWeightsCast");
    MarkCompressedWeightsCast();
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
