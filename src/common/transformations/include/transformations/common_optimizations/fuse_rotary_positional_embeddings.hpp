// Copyright (C) 2018-2025 Intel Corporation
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
class TRANSFORMATIONS_API RoPEFusionGPTJ;
class TRANSFORMATIONS_API RoPEFusionChatGLM;
class TRANSFORMATIONS_API RoPEFusionChatGLMHF;
class TRANSFORMATIONS_API RoPEFusionQwen;
class TRANSFORMATIONS_API RoPEFusionIOSlicing;
class TRANSFORMATIONS_API RoPEFusionPreprocess;
class TRANSFORMATIONS_API RoPEFusionCosSinPreprocess;
class TRANSFORMATIONS_API RoPEShareCosSin;
class TRANSFORMATIONS_API RoPEFusionVIT3D;

}  // namespace pass
}  // namespace ov

class ov::pass::RoPEFusionGPTNEOX : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPEFusionGPTNEOX");
    RoPEFusionGPTNEOX();
};

class ov::pass::RoPEFusionFlux : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPEFusionFlux");
    RoPEFusionFlux();
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

class ov::pass::RoPEFusionVIT3D : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPEFusionVIT3D");
    RoPEFusionVIT3D();
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
