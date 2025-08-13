// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pattern/multi_matcher.hpp"
namespace ov ::npuw ::patterns ::pre_compute {

class RopePatternDesc {
protected:
    ov::pass::MultiMatcher::Callback init_cb;

public:
    std::shared_ptr<ov::Node> matched_inv_freq;
    std::shared_ptr<ov::Node> matched_position_ids;
    std::shared_ptr<ov::Node> matched_sin;
    std::shared_ptr<ov::Node> matched_cos;
    std::shared_ptr<ov::Node> matched_concat;

    std::function<void()> transform_cb;

    ov::pass::MultiMatcher::Callback make_matcher_callback() {
        return [this](const auto& matches) {
            init_cb(matches);
            transform_cb();
        };
    }
};

class RopePatternLLama2 : public RopePatternDesc {
    ov::pass::MultiMatcher matcher;

public:
    using RopePatternDesc::transform_cb;
    RopePatternLLama2();
    bool run_on_model(const std::shared_ptr<ov::Model>& m) {
        return matcher.run_on_model(m);
    }
};

class RopeCacheMatcher {
public:
    RopeCacheMatcher(const uint32_t max_prompt_len, const std::shared_ptr<ov::Model>& m);
};

// TODO: not used - only in tests
// matches inverse freq tensor
class RopeInverseFreq {
public:
    using CPtr = std::shared_ptr<ov::op::v0::Constant>;
    using Results = std::reference_wrapper<std::vector<CPtr>>;

    RopeInverseFreq(Results need_freq_consts, const std::shared_ptr<ov::Model>& m);
};

class RopeCache : public ov::pass::ModelPass {
    const uint32_t m_max_prompt_len = 0;

public:
    OPENVINO_MODEL_PASS_RTTI("npuw::patterns::precompute::Rope");
    /*
     * Rope cache is NPUW  pass that removes sin/cos subgraph and replaces it with corresponding LUT/gather operations
     */
    explicit RopeCache(const uint32_t max_prompt_len) : m_max_prompt_len(max_prompt_len) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace ov::npuw::patterns::pre_compute
