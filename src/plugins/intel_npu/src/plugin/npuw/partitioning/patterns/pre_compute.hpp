// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov :: npuw :: patterns :: pre_compute {

class RPEPattern {
    std::function<bool(ov::pass::pattern::Matcher&, RPEPattern &)>callback_keep;
public:
    RPEPattern();
    std::shared_ptr<ov::Node> pattern;
    bool callback(ov::pass::pattern::Matcher& m) {
        return callback_keep(m, *this);
    }

    std::shared_ptr<ov::Node> matched_inv_freq;
    std::shared_ptr<ov::Node> matched_position_ids;
    std::shared_ptr<ov::Node> matched_sin;
    std::shared_ptr<ov::Node> matched_cos;
};
class SinCosLLama2 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::precompute::SinCosLLama2");

    SinCosLLama2(const uint32_t max_prompt_len = 1024);
};


// keep only inverse freq results - TODO: adapt for any pattern
class RopeInverseFreq : public ov::pass::MatcherPass {
    public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::precompute::InverseFreq");

    using CPtr = std::shared_ptr<ov::op::v0::Constant>;
    using Results = std::reference_wrapper<std::vector<CPtr>>;

    RopeInverseFreq(Results need_freq_consts);
};

class RopeCache : public ov::pass::ModelPass {
    const bool m_build_cache = false;
    const uint32_t m_max_prompt_len = 0;

public:
    OPENVINO_MODEL_PASS_RTTI("npuw::patterns::precompute::Rope");
    /*
     * Rope cache is NPUW  pass that removes sin/cos subgraph and replaces it with corresponding LUT/gather operations
     */
    RopeCache(const bool build_cache = true, const uint32_t max_prompt_len = 1024)
        : m_build_cache(build_cache), m_max_prompt_len(max_prompt_len) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

//
} // namespace ov
//