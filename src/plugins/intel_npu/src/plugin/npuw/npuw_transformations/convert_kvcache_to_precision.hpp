// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>
#include <string>

#include "kv_cache_compressed.hpp"

#include "openvino/core/model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::npuw {

ov::element::Type optimize_kv_cache_storage(const std::shared_ptr<ov::Model>& model);

/// Parsed NPUW_LLM_KV_CACHE_COMPRESSION: comma-separated parts, one per cache, each a colon-separated list of
/// attributes in any order, e.g. "KEY:ASYM:BY_TOKEN:INT4,VALUE:SYM". Every attribute is matched by name:
///   KEY | VALUE        - which cache the part configures (required, at most one part per cache)
///   SYM | ASYM         - quantization type (default SYM)
///   BY_TOKEN           - granularity (default and the only supported one; BY_CHANNEL is rejected)
///   INT4 | INT8        - storage precision (default: the KV-cache precision hint for the key, i8 for the value)
struct KVCacheCompressionSetting {
    struct Cache {
        KVCacheCompressionConfig::QuantizationType type = KVCacheCompressionConfig::QuantizationType::Symmetric;
        std::string granularity = "BY_TOKEN";
        std::optional<ov::element::Type> precision;  // empty: derived from the KV-cache precision hint
    };
    Cache key;
    Cache value;
};

KVCacheCompressionSetting parse_kv_cache_compression_config(const std::string& config_str);

/// Final compression parameters (storage precision and quantization type of the key and the value cache) for
/// an integer KV-cache precision hint (i8 / u8).
KVCacheCompressionParams resolve_kv_cache_compression(const ov::element::Type& kv_cache_precision,
                                                      const KVCacheCompressionSetting& setting);

class ConvertKVCacheToPrecision : public ov::pass::ModelPass {
    ov::element::Type m_lp_type;
    bool m_v_tensors_transposed = false;

    std::string m_kv_cache_compression_config;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::ConvertKVCacheToPrecision");
    explicit ConvertKVCacheToPrecision(const ov::element::Type lptype,
                                       bool v_tensors_transposed = false,
                                       const std::string& kv_cache_compression_config = "");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
