// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface MarkApproximateSoftmaxExp
 * @brief Marks the Exp that SoftmaxDecomposition emits, so that jit_exp_emitter evaluates it with a
 *        degree-1 polynomial instead of the accurate one. Registered only when the caller asked for
 *        the approximation via ov::intel_cpu::snippets_approximate_softmax_exp.
 *
 *        Only that Exp is marked. The accuracy of the approximation was only ever characterised on
 *        a value that is divided by a sum of itself along the reduction axis, so any other Exp --
 *        including the one jit_erf_emitter owns -- keeps the accurate path.
 *
 *        The pass must run directly after SoftmaxDecomposition, while the pattern is still exactly
 *        as it was emitted. Later passes obscure it: PropagatePrecision puts ConvertSaturation on
 *        either leg, and MulAddToFMA folds the normalising Multiply into a FusedMulAdd whenever its
 *        consumer is an Add. Matching after either of those would fail silently and simply leave
 *        the approximation switched off.
 * @ingroup snippets
 */
class MarkApproximateSoftmaxExp : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MarkApproximateSoftmaxExp");
    MarkApproximateSoftmaxExp();
};

}  // namespace ov::intel_cpu::pass
