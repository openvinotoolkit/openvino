// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "openvino/pass/pattern/multi_matcher.hpp"
#include "openvino/runtime/tensor.hpp"

// Forward declaration - LongRopeHostLut::serialize() below only needs the type name.
namespace ov ::npuw ::orc {
class Stream;
}  // namespace ov::npuw::orc

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

// The LongRoPE short/long-factor cos/sin tables used by the unrotated-KV mitigation
// (see CacheRawKeyPattern in pre_compute.cpp). Rows are indexed by absolute position
// (row i == position i); columns [0, rotary_ndims) are the rotate_half-style values
// (second half mirrors the first), columns [rotary_ndims, head_dim) are identity
// (cos=1, sin=0) for the passthrough part of the head. The layout therefore matches
// the npuw_lr_full_cos/npuw_lr_full_sin model Parameters 1:1, so filling them at
// runtime is a plain copy.
//
// SINGLE SOURCE OF TRUTH. For a variant where the rewrite applies, these host-owned
// buffers are the ONLY place the RoPE coefficients exist: the rewrite also rewires the
// Q-side cos/sin to a tail slice of the very same Parameters, so no cos/sin Constant
// (and no short/long Select) is emitted into the graph at all. Nothing here aliases
// compiled-blob memory - a driver is free to repack or otherwise transform the
// constants it is given, so host data must never be assumed to still match them.
//
// Coefficients are computed in f32 (std::cos/std::sin) and stored as f16, which is
// also the element type of the two model Parameters.
//
// One instance is kept per compiled model variant (prefill / each generate kvcache
// size) on LLMCompiledModel. Only max_len/rotary_ndims/head_dim and the two
// inverse-frequency arrays are written to the blob; rebuild_tables() regenerates the
// tables byte-for-byte on import (see serialize()).
struct LongRopeHostLut {
    size_t max_len = 0;       // rows; == the variant's full K context length
    size_t rotary_ndims = 0;  // rotary columns; the rest, up to head_dim, are identity
    size_t head_dim = 0;      // row width; == the npuw_lr_full_cos/sin last dim

    // f16, shape [1, max_len, head_dim].
    ov::Tensor cos_short;
    ov::Tensor sin_short;
    ov::Tensor cos_long;
    ov::Tensor sin_long;

    std::vector<float> inv_freq_short;  // rotary_ndims/2 entries
    std::vector<float> inv_freq_long;   // rotary_ndims/2 entries

    bool is_valid() const {
        return max_len > 0 && rotary_ndims > 0 && head_dim >= rotary_ndims && static_cast<bool>(cos_short) &&
               static_cast<bool>(sin_short) && static_cast<bool>(cos_long) && static_cast<bool>(sin_long);
    }

    // (Re)builds the four tables from max_len/rotary_ndims/head_dim and the two
    // inverse-frequency arrays. Used both at compile time and on blob import.
    void rebuild_tables();

    // Blob export/import - see the note above about what actually goes to the blob.
    void serialize(ov::npuw::orc::Stream& stream);
};

class RopeCacheMatcher {
public:
    // When cache_raw_key_at_attention is true and a LongRopePatternPhi_v5 match is
    // found, the K-cache (past_key_values.*.key / present.*.key) is rewritten to
    // store the RAW (pre-RoPE) key instead of the rotated one, and RoPE is applied
    // to the key right before it's consumed by attention (see CacheRawKeyPattern /
    // applyCacheRawKeyAtAttention in pre_compute.cpp). This avoids the LongRoPE
    // short/long-factor mismatch between keys cached under different regimes. When
    // this happens, out_lut (if non-null) receives the LongRopeHostLut handle - the
    // layout plus the (shared, not duplicated) cos/sin tables - the runtime needs to
    // fill the two new npuw_lr_full_cos/npuw_lr_full_sin Parameters this rewrite adds.
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
