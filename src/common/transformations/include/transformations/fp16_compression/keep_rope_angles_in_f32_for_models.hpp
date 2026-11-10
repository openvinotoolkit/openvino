// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Keeps the RoPE angle computation (positions -> angles -> Sin/Cos) in f32 under f16 / bf16 inference for an
 * explicit list of models.
 *
 * Some models compute their rotary position angles in f32 by design even when the rest of the model runs in f16 or
 * bf16, because the angles reach thousands of radians, where the f16 / bf16 spacing is a sizable fraction of the
 * period. Lowering that chain makes the model diverge from its own reduced-precision reference.
 *
 * The transformation is opt-in per model: each supported model is recognized by the graph signature of its own RoPE
 * angle chain, so models that are not listed are never affected. To support another model, add an enumerator to
 * Model, a matcher that recognizes its angle chain, and a case in the constructor.
 *
 * On match, the Sin/Cos pair and the whole angle sub-graph up to the model inputs (Parameters and ShapeOf sub-graphs
 * excluded) are marked with disable_conversion(f16), which ConvertPrecision honors under f16. The Sin/Cos pair is also
 * marked with disable_conversion(f32, bf16): Sin/Cos survive later fusions of the angle chain, and the CPU plugin
 * keeps a continuous f32 region around them under bf16. Run it before ConvertPrecision and before fusions.
 */
class TRANSFORMATIONS_API KeepRopeAnglesInF32ForModels : public ov::pass::GraphRewrite {
public:
    enum class Model {
        LTX2,  // Lightricks LTX-2 / LTX-2.3 (diffusers LTX2AudioVideoRotaryPosEmbed)
    };

    OPENVINO_GRAPH_REWRITE_RTTI("KeepRopeAnglesInF32ForModels");
    explicit KeepRopeAnglesInF32ForModels(const std::vector<Model>& models);
};

/**
 * @ingroup ov_transformation_common_api
 * @brief LTX-2 matcher of KeepRopeAnglesInF32ForModels.
 *
 * Matches the fractional RoPE of diffusers LTX2AudioVideoRotaryPosEmbed, where the positions are normalized to [0, 1]
 * and mapped to [-1, 1] before the frequency product:
 *
 *   grid -> Multiply(2) -> Add(-1) -> Multiply(freqs) -> Transpose|Reshape -> [Reshape] -> Sin
 *                                                                                       \-> Cos
 *
 * with freqs = theta ^ linspace(0, 1, N) * pi / 2, either as that sub-graph or constant-folded (freqs[0] == pi / 2).
 */
class TRANSFORMATIONS_API KeepLTX2RopeAnglesInF32 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("KeepLTX2RopeAnglesInF32");
    KeepLTX2RopeAnglesInF32();
};

}  // namespace pass
}  // namespace ov
