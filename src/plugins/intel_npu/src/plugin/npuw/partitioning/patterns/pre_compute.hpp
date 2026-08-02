// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

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
    bool duplicate_freqs = false;

    std::function<void()> transform_cb;

    ov::pass::MultiMatcher::Callback make_matcher_callback() {
        return [this](const auto& matches) {
            init_cb(matches);
            transform_cb();
        };
    }
};

class LongRopePatternDesc : public RopePatternDesc {
public:
    std::shared_ptr<ov::Node> matched_inv_freq_long;
    std::shared_ptr<ov::Node> matched_cond;
    std::shared_ptr<ov::Node> max_pos_id;
};

class LongRopev5PatternDesc : public RopePatternDesc {
public:
    std::shared_ptr<ov::Node> matched_short_factor;
    std::shared_ptr<ov::Node> matched_long_factor;
    std::shared_ptr<ov::Node> matched_context_limit;
    std::shared_ptr<ov::Node> matched_cond;
    std::shared_ptr<ov::Node> max_pos_id;
    std::shared_ptr<ov::Node> matched_multiply_const;
    std::shared_ptr<ov::Node> matched_power_const;
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

class LongRopePatternPhi : public LongRopePatternDesc {
    ov::pass::MultiMatcher matcher;

public:
    using LongRopePatternDesc::transform_cb;
    LongRopePatternPhi();
    bool run_on_model(const std::shared_ptr<ov::Model>& m) {
        return matcher.run_on_model(m);
    }
};

class LongRopePatternPhi_v5 : public LongRopev5PatternDesc {
    ov::pass::MultiMatcher matcher;

public:
    using LongRopev5PatternDesc::transform_cb;
    LongRopePatternPhi_v5();
    bool run_on_model(const std::shared_ptr<ov::Model>& m) {
        return matcher.run_on_model(m);
    }
};

// Runs LongRopePatternPhi_v5 on the model and, if matched, extracts the
// original_max_position_embeddings context limit constant it captures.
// Returns std::nullopt if the pattern doesn't match.
std::optional<uint64_t> extract_phi_v5_longrope_context_limit(const std::shared_ptr<ov::Model>& model);

// Host-side (non-graph) copy of the LongRoPE short/long-factor cos/sin LUT used
// to rotate the cached-raw K at attention time (see CacheRawKeyPattern in
// pre_compute.cpp). Rows are indexed by absolute position (row i == position i).
// Columns [0, rotary_ndims) hold the actual rotate_half-style cos/sin values
// (duplicated/mirrored the same way makeCosSinCache does); columns
// [rotary_ndims, head_dim) are identity padding (cos=1, sin=0) for the
// passthrough (non-rotary) part of the head, so the two tables can be
// multiplied directly against the FULL (unsplit) raw K tensor.
//
// This is computed once per compiled model variant (prefill / each generate
// kvcache size) and stored on LLMCompiledModel; at runtime it's copied (as a
// prefix - see the "row i == position i" invariant) into the model's own
// npuw_lr_full_cos/npuw_lr_full_sin input tensors (see llm_infer_request.cpp).
// The two inv_freq arrays are kept around so the runtime can also compute the
// LUT-uncovered tail rows (the current call's own query tokens - their
// absolute position isn't known at compile time) with a few cos/sin evals.
struct LongRopeHostLut {
    size_t max_len = 0;
    size_t rotary_ndims = 0;
    size_t head_dim = 0;
    std::vector<float> cos_short;  // max_len * head_dim, row-major
    std::vector<float> sin_short;
    std::vector<float> cos_long;
    std::vector<float> sin_long;
    std::vector<float> inv_freq_short;  // rotary_ndims/2 entries
    std::vector<float> inv_freq_long;   // rotary_ndims/2 entries

    bool is_valid() const {
        return max_len > 0 && head_dim > 0 && !cos_short.empty();
    }
};

class RopeCacheMatcher {
public:
    // When cache_raw_key_at_attention is true and a LongRopePatternPhi_v5 match is
    // found, the K-cache (past_key_values.*.key / present.*.key) is rewritten to
    // store the RAW (pre-RoPE) key instead of the rotated one, and RoPE is applied
    // to the key right before it's consumed by attention (see cache_raw_key.cpp/hpp
    // - applyCacheRawKeyAtAttention). This avoids the LongRoPE short/long-factor
    // mismatch between keys cached under different regimes. When this happens,
    // out_lut (if non-null) receives the host-side LUT data (see LongRopeHostLut)
    // needed to populate the two new npuw_lr_full_cos/npuw_lr_full_sin Parameters
    // this rewrite adds to the model, at runtime.
    RopeCacheMatcher(const uint32_t max_prompt_len,
                     const std::shared_ptr<ov::Model>& m,
                     const std::string& longrope_input_name,
                     bool cache_raw_key_at_attention = false,
                     LongRopeHostLut* out_lut = nullptr);
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
    std::string m_longrope_input_name;
    bool m_cache_raw_key_at_attention = false;
    LongRopeHostLut m_host_lut;

public:
    OPENVINO_MODEL_PASS_RTTI("npuw::patterns::precompute::Rope");
    /*
     * Rope cache is NPUW  pass that removes sin/cos subgraph and replaces it with corresponding LUT/gather operations
     */
    explicit RopeCache(const uint32_t max_prompt_len,
                       const std::string& longrope_input_name,
                       bool cache_raw_key_at_attention = false)
        : m_max_prompt_len(max_prompt_len),
          m_longrope_input_name(longrope_input_name),
          m_cache_raw_key_at_attention(cache_raw_key_at_attention) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

    // Valid (is_valid() == true) only when cache_raw_key_at_attention was set and
    // the LongRopePatternPhi_v5 pattern matched - see LongRopeHostLut.
    const LongRopeHostLut& host_lut() const {
        return m_host_lut;
    }
};
// NOLINTNEXTLINE(readability/namespace)
}  // namespace ov::npuw::patterns::pre_compute
