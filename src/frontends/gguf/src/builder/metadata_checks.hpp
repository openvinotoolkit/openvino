// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include "openvino/core/except.hpp"
#include "openvino/frontend/gguf/builder/metadata.hpp"

namespace ov::frontend::gguf {

inline void require_positive_finite(double value, const std::string& what) {
    OPENVINO_ASSERT(std::isfinite(value) && value > 0, "[GGUF] ", what, " must be finite and positive");
}

// `what` names the consumer in the error, e.g. "Mamba2 metadata".
inline int64_t require_positive_int(const GgufMetadata& metadata, const std::string& key, const std::string& what) {
    const auto value = metadata.get_int(key);
    OPENVINO_ASSERT(value && *value > 0 && *value <= std::numeric_limits<int>::max(),
                    "[GGUF] ",
                    what,
                    " requires a positive integer '",
                    key,
                    "'");
    return *value;
}

inline float require_positive_float(const GgufMetadata& metadata, const std::string& key, const std::string& what) {
    const auto value = metadata.get_float(key);
    OPENVINO_ASSERT(value && std::isfinite(*value) && *value > 0,
                    "[GGUF] ",
                    what,
                    " requires a positive finite number '",
                    key,
                    "'");
    return static_cast<float>(*value);
}

}  // namespace ov::frontend::gguf
