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
class TRANSFORMATIONS_API RoPEFusionQwen;
class TRANSFORMATIONS_API RoPEFusionIOSlicing;
class TRANSFORMATIONS_API RoPEFusionPreprocess;
class TRANSFORMATIONS_API RoPEFusionCosSinPreprocess;
class TRANSFORMATIONS_API RoPEShareCosSin;

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
    RoPEFusionChatGLM(int split_output_id, const bool support_2d_rope = false);
};

class ov::pass::RoPEFusionQwen : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RoPEFusionQwen");
    RoPEFusionQwen(int split_output_id);
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

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses special sub-graph into an internal Rotary Positional Embedding operation
 */
class ov::pass::RoPEFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("RoPEFusion");
    RoPEFusion(bool support_2d_rope = false) {
        add_matcher<ov::pass::RoPEFusionFlux>();
        add_matcher<ov::pass::RoPEFusionGPTNEOX>();
        add_matcher<ov::pass::RoPEFusionGPTJ>();
        // optional heads & tails are fused in separate matcher pass,
        // after RoPENode has been created.
        add_matcher<ov::pass::RoPEFusionCosSinPreprocess>();
        add_matcher<ov::pass::RoPEFusionIOSlicing>();
        add_matcher<ov::pass::RoPEFusionPreprocess>();

        add_matcher<ov::pass::RoPEFusionChatGLM>(0);
        add_matcher<ov::pass::RoPEFusionChatGLM>(1);
        if (support_2d_rope) {
            add_matcher<ov::pass::RoPEFusionChatGLM>(0, true);
            add_matcher<ov::pass::RoPEFusionChatGLM>(1, true);
        }

        add_matcher<ov::pass::RoPEFusionQwen>(0);
        add_matcher<ov::pass::RoPEFusionQwen>(1);

        add_matcher<ov::pass::RoPEShareCosSin>();
    }
};
