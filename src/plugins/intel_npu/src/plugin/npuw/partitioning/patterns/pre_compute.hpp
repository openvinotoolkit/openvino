// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov :: npuw :: patterns :: pre_compute {

class RopePatternDesc {
protected:
    ov::matcher_pass_callback verifier_cb;
public:
    std::shared_ptr<ov::Node> pattern;
    std::shared_ptr<ov::Node> matched_inv_freq;
    std::shared_ptr<ov::Node> matched_position_ids;
    std::shared_ptr<ov::Node> matched_sin;
    std::shared_ptr<ov::Node> matched_cos;
    //TODO: can be moved to private and init in ctor
    ov::matcher_pass_callback callback;

    ov::matcher_pass_callback make_matcher_callback() {
        return [this](pass::pattern::Matcher& m) {
            if (!verifier_cb(m)) {
                return false;
            }
            return callback(m);
        };
    }
};

class RopePatternLLama2 : public RopePatternDesc {
public:
    //using RopePatternDesc::callback;
    RopePatternLLama2();
};

class RopeCacheMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::precompute::SinCosLLama2");
    RopeCacheMatcher(const uint32_t max_prompt_len);
};

// TODO: not used - only in tests
// matches inverse freq tensor
class RopeInverseFreq : public ov::pass::MatcherPass {
    public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::precompute::InverseFreq");

    using CPtr = std::shared_ptr<ov::op::v0::Constant>;
    using Results = std::reference_wrapper<std::vector<CPtr>>;

    RopeInverseFreq(Results need_freq_consts);
};

class RopeCache : public ov::pass::ModelPass {
    const uint32_t m_max_prompt_len = 0;

public:
    OPENVINO_MODEL_PASS_RTTI("npuw::patterns::precompute::Rope");
    /*
     * Rope cache is NPUW  pass that removes sin/cos subgraph and replaces it with corresponding LUT/gather operations
     */
    RopeCache(const uint32_t max_prompt_len)
        : m_max_prompt_len(max_prompt_len) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

//
} // namespace ov
//