// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/backward_graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/**
 * @brief Fuses an L2 normalization over the last axis, as ONNX LpNormalization(p=2) decomposes it, into
 * ov::op::internal::RMS.
 *
 * Over N elements ReduceSum = N * ReduceMean, so
 *     x / Sqrt(ReduceSum(x^2) + eps) * gamma == RMS(x, gamma / sqrt(N), eps / N).
 * RMSFusion does not take this form: it looks for ReduceMean and requires gamma to be a Constant, while here gamma
 * may be computed at run time (e.g. a dequantized weight fed as a model input).
 *
 * Before:                                           After:
 *
 *     x ──────────────┬──────┐                          x      gamma
 *     │ [Convert]     │      │                          │        │
 *   Power(2)          │      │                          │   Multiply(1/sqrt(N))
 *     │               │      │                          │        │
 *   ReduceSum(-1, keep_dims) │                          └───┬────┘
 *     │               │      │                             RMS(eps / N, axis = -1)
 *   Add(eps)          │      │
 *     │               │      │
 *   Sqrt              │      │
 *     └─────► Divide ◄┘      │
 *               │            │
 *            Multiply ◄── gamma
 *               │
 *           [Convert]   (absorbed only when it is the sole consumer; RMS then writes its type)
 *
 * Applied when:
 *  - the reduction is over the last axis of a rank <= 5 input (the GPU RMS kernel has bfyx/bfzyx implementations
 *    only) and that axis is static, since N goes into gamma and eps;
 *  - N / 8 <= max_work_group_size, the limit RMSFusion applies on GPU (one work group per row);
 *  - gamma has a static shape {..., 1, N} with no more dims than x, so it scales the normalized axis only;
 *  - eps is a uniform Constant within the finite f32 range and every binary eltwise op uses numpy auto_broadcast.
 */
class L2NormToRMSLastAxis : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("L2NormToRMSLastAxis");
    explicit L2NormToRMSLastAxis(size_t max_work_group_size);
};

/**
 * @brief Fuses an L2 normalization over the channel axis, as torch's F.normalize(x, dim=1) followed by an affine
 * scale exports it into ov::op::internal::RMS on axis 1.
 *
 * With C channels ||x||_2 = sqrt(C) * rms(x) and max(||x||_2, eps) = sqrt(C * max(mean(x^2), eps^2 / C)), so
 *     x / max(||x||_2, eps) * s * gamma ~= RMS(x, s * gamma / sqrt(C), eps^2 / C, axis = 1).
 * The only difference is how eps enters (a floor instead of an addend): for 0 < ||x||_2 < eps the results differ by a
 * relative ~eps^2 / (2 ||x||_2^2), so the pass requires eps small enough for that to vanish in the element type (below).
 * This replaces the reduction + several full-tensor eltwise ops with one kernel that reads x once.
 *
 * Before:                                                  After:
 *
 *     x ─────────────────────────────┐                         x      [s] [gamma]
 *     │                              │                         │        │    │
 *   ReduceL2(axis 1, keep_dims)      │                         │    gamma' = Broadcast(Reshape(
 *     │                              │                         │      1/sqrt(C) [* s] [* gamma], [-1]), {C})
 *   Clamp(eps, clip)                 │                         │        │   (constant-folded when possible)
 *     | Maximum(eps) -> [Minimum(clip)]                        └───┬────┘
 *     │                              │                            RMS(eps^2 / C, axis = 1)
 *   [Broadcast]                      │
 *     │                              │
 *     └──────────► Divide(x, norm) ◄─┘   | Multiply(x, Power(norm, -1))
 *                    │
 *                 [Multiply(s)]          s:     scalar or per-channel
 *                    │
 *                 [Multiply(gamma)]      gamma: scalar or per-channel, not necessarily a Constant
 *                    │
 *                 [Add(0)]               all-zero Constant
 *
 * [..] marks optional nodes; Constants may sit behind a (decompression) Convert.
 *
 * Applied when:
 *  - x has rank 4 or 5 (RMS maps axis 1 to the GPU feature axis for those ranks only) and a static channel
 *    dimension C (it goes into gamma and eps); other dimensions may be dynamic;
 *  - the reduction is over axis 1 only, with keep_dims, and every matched node but the last feeds nothing else
 *    (otherwise it would stay alive next to RMS);
 *  - eps is a scalar in [0, d / 1024], d the smallest positive (denormal) value of the element type of the norm
 *    (f16: ~5.8e-11, so the usual 1e-12 qualifies; f32/bf16: practically only 0). A nonzero ||x||_2 is at least d, so
 *    the floor and the addend then agree to 2^-21 relative, and ||x||_2 == 0 gives 0 either way;
 *  - the clip, if any, is not below the largest finite value of the element type it is applied in, so it cannot
 *    change a finite result (f16, bf16 and f32 only);
 *  - every binary eltwise op of the chain uses numpy auto_broadcast;
 *  - per-channel factors broadcast along axis 1 only and no op of the chain changes the shape of x.
 */
class L2NormToRMSChannelAxis : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("L2NormToRMSChannelAxis");
    L2NormToRMSChannelAxis();
};

/**
 * @brief Runs L2NormToRMSLastAxis and L2NormToRMSChannelAxis bottom-up: both patterns end in optional nodes, and
 * backward traversal reaches the outermost of them first, so the longest chain is fused.
 */
class L2NormToRMS : public ov::pass::BackwardGraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("L2NormToRMS");
    explicit L2NormToRMS(size_t max_work_group_size) {
        add_matcher<L2NormToRMSLastAxis>(max_work_group_size);
        add_matcher<L2NormToRMSChannelAxis>();
    }
};

}  // namespace ov::intel_gpu
