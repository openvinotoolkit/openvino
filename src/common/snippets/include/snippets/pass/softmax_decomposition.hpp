// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::snippets::pass {

/**
 * @interface SoftmaxDecomposition
 * @brief Decomposes Softmax to a range of low-level operations.
 *
 *        With defer_normalization, and where the Softmax feeds an integer matmul across a
 *        quantizer, the reciprocal of the row sums is applied to the matmul's dequantized output
 *        instead of to the Softmax, so that the quantizer sees exp(s - rowmax) rather than the
 *        normalized probabilities. The two forms agree in exact arithmetic. In quantized arithmetic
 *        they do not: the row sums are at least 1, so the deferred operand occupies at least as
 *        much of the quantization grid, and the rescale divides the resulting error back down by
 *        the same factor. The error bound is therefore smaller, but the result is not the one the
 *        source graph specifies, and on a grid the normalized probabilities barely use the two can
 *        differ by more than they agree. That is why this is off by default and has to be asked
 *        for by a consumer that has measured the trade-off on its own models.
 *
 *        When it is asked for, the rewrite is still emitted only where the pass has established on
 *        the graph that each step between the Softmax and the matmul either commutes with a
 *        positive per-row scale or is the quantization step itself, applied to an operand a scale
 *        has already spread over the grid; and that neither the quantizer's range nor the matmul's
 *        i32 accumulator is narrower than the deferred operand needs.
 * @ingroup snippets
 */
class SoftmaxDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::SoftmaxDecomposition");
    explicit SoftmaxDecomposition(bool defer_normalization = false);
};

}  // namespace ov::snippets::pass
