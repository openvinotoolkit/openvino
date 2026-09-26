// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/openvino.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace npuw {

namespace online {
class Snapshot;  // Forward declaration
}  // namespace online

namespace patterns {

// Note: the patterns below are only utilized by the online partitioner
namespace attn {

class SDPA : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::attn::SDPA");
    static constexpr const char* pattern_name() {
        return "SDPA";
    }
    static constexpr const char* isolation_tag() {
        return "attn";
    }
    static constexpr const char* group_name() {
        return "attn";
    }
    SDPA(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class SDPADecomposed : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::attn::SDPADecomposed");
    static constexpr const char* pattern_name() {
        return "SDPADecomposed";
    }
    static constexpr const char* isolation_tag() {
        return "attn";
    }
    static constexpr const char* group_name() {
        return "attn";
    }
    SDPADecomposed(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class QuantizedSDPAWithGlobalMask : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::attn::QuantizedSDPAWithGlobalMask");
    static constexpr const char* pattern_name() {
        return "QuantizedSDPAWithGlobalMask";
    }
    static constexpr const char* isolation_tag() {
        return "attn";
    }
    static constexpr const char* group_name() {
        return "attn";
    }
    QuantizedSDPAWithGlobalMask(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot,
                                const std::string& isol_tag);
};

// Matches decomposed SDPA pattern where past KV cache inputs have been converted
// to integer precision (i8/u8) with dynamic dequantization nodes inserted by
// ConvertKVCacheToPrecision. The dequantization chain is:
//   [any_input] → Subtract(zp) → Multiply(scale) → Concat
// instead of the original:
//   Convert → Concat
class SDPACompressed : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::attn::SDPACompressed");
    static constexpr const char* pattern_name() {
        return "SDPACompressed";
    }
    static constexpr const char* isolation_tag() {
        return "attn";
    }
    static constexpr const char* group_name() {
        return "attn";
    }
    SDPACompressed(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

}  // namespace attn

namespace regularize {

inline constexpr const char* PRESERVE_CONCAT_AXIS_GATHERS_RT_KEY = "npuw_preserve_concat_axis_gathers";

class AttentionBroadcast : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::attn::AttentionBroadcast");
    explicit AttentionBroadcast(bool preserve_concat_axis_gathers = false);
};

class AttentionBroadcast2 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::attn::AttentionBroadcast2");
    explicit AttentionBroadcast2(bool preserve_concat_axis_gathers = false);
};

class AttentionBroadcast3 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::attn::AttentionBroadcast3");
    explicit AttentionBroadcast3(bool preserve_concat_axis_gathers = false);
};

class AttentionBroadcast4 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::regularize::AttentionBroadcast4");
    AttentionBroadcast4();
};

class SeparateKVCache : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::regularize::SeparateKVCache");
    SeparateKVCache();
};

class ShapeOfParameter : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::attn::ShapeOfParameter");
    ShapeOfParameter();
};

// Preserves Concat-axis Gathers for chunked prefill while folding other bounded shape values.
class ShapeOfConcat : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::attn::ShapeOfConcat");
    explicit ShapeOfConcat(bool preserve_concat_axis_gathers = false);
};

class RegularizeSDPA : public ov::pass::ModelPass {
    bool m_run_broadcast_pattern = false;
    bool m_preserve_concat_axis_gathers = false;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::RegularizeSDPA");
    explicit RegularizeSDPA(bool run_broadcast_pattern, bool preserve_concat_axis_gathers = false)
        : m_run_broadcast_pattern(run_broadcast_pattern),
          m_preserve_concat_axis_gathers(preserve_concat_axis_gathers){};

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace regularize

}  // namespace patterns
}  // namespace npuw
}  // namespace ov
