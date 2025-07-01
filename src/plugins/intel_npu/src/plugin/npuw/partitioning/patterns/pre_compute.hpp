// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov :: npuw :: patterns :: pre_compute {

class SinCosLLama2 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::precompute::SinCosLLama2");
    SinCosLLama2();
};
class RopeCache : public ov::pass::ModelPass {
    bool m_build_cache = false;
public:
    OPENVINO_MODEL_PASS_RTTI("npuw::patterns::precompute::Rope");
    /*
     * Rope fusion is openvino pass that simplifies sin/cos subgraph detection
     */
    RopeCache(bool build_cache = true, bool run_rope_fudion = false)
        : m_build_cache(build_cache) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

//
} // namespace ov
//