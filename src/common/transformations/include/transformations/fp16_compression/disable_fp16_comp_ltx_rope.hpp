// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {
class TRANSFORMATIONS_API DisableFP16CompForLtxRope;
class TRANSFORMATIONS_API DisableFP16CompForLtxVideoRopePattern;
class TRANSFORMATIONS_API DisableFP16CompForLtx2RopePattern;
}  // namespace ov::pass

/**
 * @ingroup ov_transformation_common_api
 * @brief Keeps the runtime-computed RoPE tables of LTX-Video and LTX-2 in the original precision.
 *
 * These models build cos/sin tables from position grids multiplied by frequencies, so the Sin/Cos
 * argument reaches ~1.6e4 rad. Low precision types quantize such values in steps larger than 2*pi
 * (f16: 8 rad, bf16: 64 rad at 1.6e4), and the frequencies amplify any rounding of the grid itself.
 * For a matched table, the chain up to Sin/Cos and the input paths of the grid and of the frequencies
 * (up to Parameters, Constants and ShapeOfs) are marked with disable_conversion(f16), honored both by
 * ConvertPrecision and by the CPU plugin's bf16 enforcement. A match whose input path reaches a MatMul
 * or a convolution depends on activations and is left unmarked.
 */
class ov::pass::DisableFP16CompForLtxRope : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("DisableFP16CompForLtxRope");
    DisableFP16CompForLtxRope();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Table with the frequencies applied before the grid shift (LTX-Video, and LTX-2 once the
 * centering and the frequency Multiply are fused).
 *
 *   grid -> Multiply(freqs) -> Add(Constant) -> [Transpose | Reshape] -> [Reshape] -> Sin/Cos
 */
class ov::pass::DisableFP16CompForLtxVideoRopePattern : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompForLtxVideoRopePattern");
    DisableFP16CompForLtxVideoRopePattern();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Table with the grid centered before the frequencies are applied (LTX-2).
 *
 *   grid -> Add(Constant) -> [Squeeze] -> [Gather] -> [Unsqueeze] -> Multiply(freqs)
 *        -> [Transpose | Reshape] -> [Reshape] -> Sin/Cos
 *
 * The layout ops are optional: the text connectors feed the angles to Sin/Cos directly.
 * The axis selection ops appear once MoveEltwiseUpThroughDataMov moves the shift above them.
 */
class ov::pass::DisableFP16CompForLtx2RopePattern : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompForLtx2RopePattern");
    DisableFP16CompForLtx2RopePattern();
};
