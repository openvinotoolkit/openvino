// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API RoPEFusion;
class TRANSFORMATIONS_API RoPEFusionGPTNEOX;
class TRANSFORMATIONS_API RoPEFusionFlux;
class TRANSFORMATIONS_API RoPEFusionPackedPairs;
class TRANSFORMATIONS_API RoPEFusionGPTJ;
class TRANSFORMATIONS_API RoPEFusionChatGLM;
class TRANSFORMATIONS_API RoPEFusionChatGLMHF;
class TRANSFORMATIONS_API RoPEFusionQwen;
class TRANSFORMATIONS_API RoPEFusionIOSlicing;
class TRANSFORMATIONS_API RoPEFusionPreprocess;
class TRANSFORMATIONS_API RoPEFusionCosSinPreprocess;
class TRANSFORMATIONS_API RoPEShareCosSin;
class TRANSFORMATIONS_API RoPEFusionGPTOSS;
class TRANSFORMATIONS_API RoPEFusionLtxVideo;
class TRANSFORMATIONS_API RoPEFusionCohere;
class TRANSFORMATIONS_API RoPEFusionSliceAssign;

}  // namespace pass
}  // namespace ov

class ov::pass::RoPEFusionGPTNEOX : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPEFusionGPTNEOX");
    RoPEFusionGPTNEOX(int rank);
};

class ov::pass::RoPEFusionFlux : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPEFusionFlux");
    RoPEFusionFlux(bool num_heads_transposed = true);
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses a RoPE applied to coordinate pairs packed in the innermost dimension into the internal
 * RoPE operation in interleaved mode.
 *
 * Before:
 *
 *   x -> reshape([batch, seq, num_heads, half_rotary_ndims, 2])
 *     -> gather(0, axis=-1) = x0, gather(1, axis=-1) = x1
 *     -> x0 * cos - x1 * sin, x0 * sin + x1 * cos
 *     -> unsqueeze(-1) each -> concat(axis=-1)
 *
 * After:
 *
 *   x -> reshape([batch, seq, num_heads, head_size]) -> RoPE(is_interleaved)
 *     -> reshape([batch, seq, num_heads, half_rotary_ndims, 2])
 *
 * The cos/sin tables [seq, 1, half_rotary_ndims] are widened to [1, seq, 1, head_size] by repeating
 * every value twice. Applied only when seq, num_heads and half_rotary_ndims are static and the tables
 * fold to constants.
 */
class ov::pass::RoPEFusionPackedPairs : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPEFusionPackedPairs");
    RoPEFusionPackedPairs();
};

class ov::pass::RoPEFusionGPTJ : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPEFusionGPTJ");
    RoPEFusionGPTJ();
};

class ov::pass::RoPEFusionChatGLM : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPEFusionChatGLM");
    RoPEFusionChatGLM(const bool support_2d_rope = false);
};

class ov::pass::RoPEFusionChatGLMHF : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPEFusionChatGLMHF");
    RoPEFusionChatGLMHF();
};

class ov::pass::RoPEFusionQwen : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPEFusionQwen");
    RoPEFusionQwen();
};

class ov::pass::RoPEFusionIOSlicing : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPEFusionIOSlicing");
    RoPEFusionIOSlicing();
};

class ov::pass::RoPEFusionPreprocess : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPEFusionPreprocess");
    RoPEFusionPreprocess();
};

class ov::pass::RoPEFusionCosSinPreprocess : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPEFusionCosSinPreprocess");
    RoPEFusionCosSinPreprocess();
};

class ov::pass::RoPEShareCosSin : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPEShareCosSin");
    RoPEShareCosSin();

private:
    std::shared_ptr<ov::Node> m_inv_freq;
    std::shared_ptr<ov::Node> m_shared_cos0;
    std::shared_ptr<ov::Node> m_shared_sin0;
    std::vector<std::shared_ptr<ov::Node>> m_shared_inputs{2, nullptr};
};

class ov::pass::RoPEFusionGPTOSS : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPEFusionGPTOSS");
    RoPEFusionGPTOSS();
};

class ov::pass::RoPEFusionLtxVideo : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPEFusionLtxVideo");
    RoPEFusionLtxVideo();
};

class ov::pass::RoPEFusionCohere : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPEFusionCohere");
    RoPEFusionCohere();
};

class ov::pass::RoPEFusionSliceAssign : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPEFusionSliceAssign");
    RoPEFusionSliceAssign();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses special sub-graph into an internal Rotary Positional Embedding operation
 */
class ov::pass::RoPEFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("RoPEFusion");
    RoPEFusion(bool support_2d_rope = false);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    bool m_support_2d_rope;
};
