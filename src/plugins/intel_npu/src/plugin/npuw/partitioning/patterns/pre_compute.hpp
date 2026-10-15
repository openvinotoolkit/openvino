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

// Forward declaration - LongRopeCosSin::serialize() below only needs the type name.
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
    std::shared_ptr<ov::Node> matched_context_limit;
    // The constant added to max(position_ids) before comparing against the context limit.
    std::shared_ptr<ov::Node> matched_cond_offset;
    std::shared_ptr<ov::Node> matched_cond;
    std::shared_ptr<ov::Node> max_pos_id;
};

class LongRopev5PatternDesc : public RopePatternDesc {
public:
    std::shared_ptr<ov::Node> matched_short_factor;
    std::shared_ptr<ov::Node> matched_long_factor;
    std::shared_ptr<ov::Node> matched_context_limit;
    // The constant added to max(position_ids) before comparing against the context limit.
    std::shared_ptr<ov::Node> matched_cond_offset;
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

// Runs the LongRoPE patterns (LongRopePatternPhi_v5, then LongRopePatternPhi) on the
// model and, if one matches, extracts the original_max_position_embeddings context
// limit constant it captures. Returns std::nullopt if neither pattern matches.
std::optional<uint64_t> extract_longrope_context_limit(const std::shared_ptr<ov::Model>& model);

// Names of the two model inputs the LongRoPE RoPE-cache
// rewrite creates for the cos/sin coefficient tables - see RopeCacheMatcher.
constexpr const char* longrope_cos_input = "npuw_lr_cos";
constexpr const char* longrope_sin_input = "npuw_lr_sin";

// Host-side LongRoPE cos/sin coefficient tables, in the exact layout of the
// npuw_lr_cos/npuw_lr_sin model inputs (f16, row i == absolute position i), so a model
// input can be pointed straight at them with set_tensor - no per-inference copy.
//
// The graph itself holds no cos/sin values for such a model: these host-owned buffers
// are the only place they exist, and the short/long-factor choice the model used to
// make with an in-graph Select is made by the host when it picks which rows to bind.
// Nothing here aliases compiled-blob memory - a driver is free to repack the constants
// it is given, so host data must never be assumed to still match them.
//
// Both regimes live in ONE tensor per trig function, [1, regimes * max_len,
// rotary_ndims]: the short-factor rows first, the long-factor rows after. Because row i
// means position i regardless of the variant, a leading slice of either half serves
// prefill and every generate variant, whatever their individual LUT lengths - so
// LLMCompiledModel keeps a single instance sized to the longest LUT in the model.
//
// The long half is omitted altogether (has_long == false) when the long regime cannot
// be reached - the context limit is at or beyond the longest LUT - or when the two
// factor sets are numerically identical, which is the case for models that declare
// LongRoPE but never actually switch (e.g. MiniCPM). Both regimes then bind the same
// rows.
//
// Only max_len/rotary_ndims/has_long and the two inverse-frequency arrays go to the
// blob; rebuild_tables() regenerates the tensors on import.
struct LongRopeCosSin {
    size_t max_len = 0;       // rows per regime; == the longest sin/cos LUT in the model
    size_t rotary_ndims = 0;  // row width; 0 means "no LongRoPE in this model"
    bool has_long = false;

    std::vector<float> inv_freq_short;  // rotary_ndims/2 entries
    std::vector<float> inv_freq_long;   // rotary_ndims/2 entries

    // f16, [1, (has_long ? 2 : 1) * max_len, rotary_ndims]. Empty until rebuild_tables().
    ov::Tensor cos;
    ov::Tensor sin;

    bool is_valid() const {
        return max_len > 0 && rotary_ndims > 0 && static_cast<bool>(cos) && static_cast<bool>(sin);
    }

    // (Re)builds the two tensors from max_len/rotary_ndims/has_long and the inverse-
    // frequency arrays. Used both at compile time and on blob import.
    void rebuild_tables();

    // Dense, non-owning views over the first lut_len rows of the requested regime.
    // The backing tensors must outlive them. Non-const because NPUW's input binding
    // path reaches for a writable data pointer.
    ov::Tensor cos_rows(size_t lut_len, bool is_long);
    ov::Tensor sin_rows(size_t lut_len, bool is_long);

    void serialize(ov::npuw::orc::Stream& stream);
};

class RopeCacheMatcher {
public:
    // On a LongRoPE match (either LongRopePatternPhi_v5 or LongRopePatternPhi) the
    // cos/sin LUTs are created as model inputs (npuw_lr_cos/npuw_lr_sin) instead of
    // Constants, and out_tables (if non-null) receives the layout and inverse
    // frequencies the runtime needs to fill them. The tensors themselves are not built
    // here - see LongRopeCosSin.
    RopeCacheMatcher(const uint32_t max_prompt_len,
                     const std::shared_ptr<ov::Model>& m,
                     LongRopeCosSin* out_tables = nullptr);
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
    LongRopeCosSin m_host_tables;

public:
    OPENVINO_MODEL_PASS_RTTI("npuw::patterns::precompute::Rope");
    /*
     * Rope cache is NPUW  pass that removes sin/cos subgraph and replaces it with corresponding LUT/gather operations
     */
    explicit RopeCache(const uint32_t max_prompt_len) : m_max_prompt_len(max_prompt_len) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

    // Layout and inverse frequencies of this model's LongRoPE LUT inputs; rotary_ndims
    // is 0 unless one of the LongRoPE patterns matched.
    const LongRopeCosSin& host_tables() const {
        return m_host_tables;
    }
};
// NOLINTNEXTLINE(readability/namespace)
}  // namespace ov::npuw::patterns::pre_compute
