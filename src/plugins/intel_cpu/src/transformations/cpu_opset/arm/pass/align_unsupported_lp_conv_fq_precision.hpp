// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

/*
 * Description:
 *     AlignUnsupportedLPConvFQPrecision aligns the output FakeQuantize's
 *     PrecisionsAttribute to match the Convolution's input precision.
 *
 *     Pattern matched: Convolution → Add (bias) → FakeQuantize
 *
 *     ARM ACL int8 convolution requires the same quantized element type (u8 or i8)
 *     for both the activation input and the post-convolution FakeQuantize output.
 *
 *     This pass runs as part of LPT MarkupOptimizations to force the output FakeQuantize
 *     PrecisionsAttribute to the Convolution's input type (if possible),
 *     enabling FakeQuantizeDecomposition to produce matching quantization types
 *     that ACL can fuse into the convolution.
 *
 * Before:
 *
 *               +-----+------+
 *               | Convolution |  (precisions attribute = u8)
 *               +-----+------+
 *                     |
 *               +-----v------+
 *               |  Add (bias)|
 *               +-----+------+
 *                     |
 *     +---------------------------------+
 *     |         FakeQuantize            |
 *     | PrecisionsAttribute = { u8, i8 }|  <-- potential mismatch with conv input,
 *     +---------------------------------+      since i8 may be chosen
 *
 * After:
 *
 *               +-----+------+
 *               | Convolution |  (precisions attribute = u8)
 *               +-----+------+
 *                     |
 *               +-----v------+
 *               |  Add (bias)|
 *               +-----+------+
 *                     |
 *     +---------------------------------+
 *     |         FakeQuantize            |
 *     | PrecisionsAttribute = { u8 }    |  <-- aligned to conv input
 *     +---------------------------------+
 */

namespace ov::intel_cpu {

class AlignUnsupportedLPConvFQPrecision : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("AlignUnsupportedLPConvFQPrecision");
    AlignUnsupportedLPConvFQPrecision();
};

}  // namespace ov::intel_cpu