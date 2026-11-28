// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

/**
 * @brief Keeps sinusoidal embeddings, such as diffusion timestep embeddings, in fp32
 *        when the CPU plugin infers in BF16.
 *
 *      angle = x[N, 1] * Constant(freq)[1, K]  ->  Sin and Cos
 *
 *  Diffusion models embed the timestep this way, with x in [0, 1000] and freq up
 *  to 1.0, so angles reach ~1000 rad. Such values have exponent 9, i.e. lie in
 *  [2^9, 2^10), and BF16 stores 7 mantissa bits, so their spacing is
 *  2^9 / 2^7 = 4: rounding errors up to 2 rad, about a third of 2*pi, which turns
 *  the high-frequency channels into noise and corrupts the embedding that
 *  conditions the whole model.
 *
 *  The node computing x, an Unsqueeze or Reshape between it and the Multiply,
 *  the Multiply, Sin and Cos are marked with @ref ov::disable_conversion(node, f16),
 *  so EnforceInferencePrecision keeps exactly these nodes in fp32, without
 *  spreading to their neighbours.
 */
class DisableBF16CompSinCosEmbedding : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableBF16CompSinCosEmbedding");
    DisableBF16CompSinCosEmbedding();
};

}  // namespace ov::intel_cpu
