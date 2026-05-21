// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

/*
 * Description:
 *     FallbackUnsupportedLPConvToFP16 rewrites low-precision ARM convolution slices that
 *     cannot stay on the intended int8 path into an explicit fp16 path.
 *
 *     The pass has two local matcher entry points:
 *       1. Convolution -> Multiply -> Add -> FakeQuantize
 *       2. Convolution -> Multiply -> Add
 *
 *     The Convolution activation can arrive either directly as u8/i8 or through an
 *     optional Convert -> Subtract zero-point dequantization chain.
 *
 *     For the FakeQuantize case, fallback is applied when either:
 *       1. The Convolution activation comes from a Subtract, which means the zero-point
 *          path is present and the int8 ACL convolution executor is not applicable.
 *       2. FakeQuantize output precision differs from the Convolution activation precision.
 *
 *     The second matcher is intentionally suffix-based: it rewrites the local
 *     Convolution -> Multiply -> Add fragment without requiring trailing Clamp or
 *     FakeQuantize nodes to be part of the matched subgraph.
 *
 *     For that local Convolution -> Multiply -> Add matcher, the matched low-precision
 *     suffix is rewritten unconditionally once it is found.
 *
 *     The rewrite moves the post-convolution dequantization scale from the output path to
 *     the Convolution weights:
 *       - activation is converted to fp16 if needed
 *       - weights are converted to fp16 if needed
 *       - Multiply scales are converted to fp16, reshaped to the weights rank, and folded
 *         into the weights through a new Multiply
 *       - the original post-convolution Multiply is removed by cloning the Convolution with
 *         fp16 activation and scaled fp16 weights
 *       - Add, optional Clamp, and optional FakeQuantize stay downstream
 *
 * Before (FakeQuantize case, zero-point activation shown):
 *
 *  +--------+     +----------+
 *  |Convert | --> | Subtract | (zero-point)    +---------------+
 *  +--------+     +-----+----+                 | Weights (i8)  |
 *                       |                      +-------+-------+
 *                       +----------+-----------+
 *                                  |
 *                            +-----v------+
 *                            | Convolution |
 *                            +------+------+
 *                                   |
 *                                   v
 *                            +-------------+
 *                            |  Multiply   |
 *                            +------+------+
 *                                   |
 *                                   v
 *                            +-------------+
 *                            |     Add     |
 *                            +------+------+
 *                                   |
 *                                   v
 *                            +-------------+
 *                            | FakeQuantize|
 *                            +------+------+
 *                                   |
 *                                   v
 *                            +-------------+
 *                            |   Result    |
 *                            +-------------+
 *
 * Before (local suffix matched without trailing Clamp/FakeQuantize anchors):
 *
 * +--------------+      +---------------+
 * | Input (u8/i8)|      | Weights (i8)  |
 * +------+-------+      +-------+-------+
 *        |                      |
 *        +----------+-----------+
 *                   |
 *             +-----v------+
 *             | Convolution |
 *             +------+------+
 *                    |
 *                    v
 *             +-------------+
 *             |  Multiply   |
 *             +------+------+
 *                    |
 *                    v
 *             +-------------+
 *             |     Add     |
 *             +------+------+
 *                    |
 *                    v
 *             +-------------------------+
 *             | [Clamp] -> [FakeQuantize]|
 *             +------------+------------+
 *                          |
 *                          v
 *                   +-------------+
 *                   |   Result    |
 *                   +-------------+
 *
 * After:
 *
 * +--------------+      +----------------+                             +--------------+
 * | Input (u8/i8)| ---> | Convert (f16)  | --------------------------> |              |
 * +--------------+      +----------------+                             | Convolution  |
 *                                                                      |              |
 * +---------------+      +----------------+      +------------------+  | (f16 act,    |
 * | Weights (i8)  | ---> | Convert (f16)  | ---> |     Multiply     |->| scaled f16   |
 * +---------------+      +----------------+      +--------+---------+  | weights)     |
 *                                                         ^            +------+-------+
 *                                                         |                   |
 * +--------------+      +----------------+      +---------+--------+          v
 * | DQ Scales    | ---> | Convert (f16)  | ---> |    Reshape       |   +------+------+
 * +--------------+      +----------------+      +------------------+   |     Add     |
 *                                                                      +------+------+
 *                                                                             |
 *                                                                             v
 *                                                                      +------+------+
 *                                                                      | Clamp (opt) |
 *                                                                      +------+------+
 *                                                                             |
 *                                                                             v
 *                                                                      +------+------+
 *                                                                      | FakeQuantize|
 *                                                                      +------+------+
 *                                                                             |
 *                                                                             v
 *                                                                      +------+------+
 *                                                                      |   Result    |
 *                                                                      +-------------+
 *
 * Note:
 *     Convert(f16) nodes are inserted only when the corresponding input is not f16.
 */

namespace ov::intel_cpu {

class FallbackUnsupportedLPConvToFP16 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FallbackUnsupportedLPConvToFP16");
    FallbackUnsupportedLPConvToFP16();
};

}  // namespace ov::intel_cpu
