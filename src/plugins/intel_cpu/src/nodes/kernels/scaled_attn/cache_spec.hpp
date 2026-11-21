// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/internal_properties.hpp"

namespace ov::intel_cpu {

struct CacheSpec {
    ov::internal::CacheQuantAlgorithm alg = ov::internal::CacheQuantAlgorithm::SCALAR;
    ov::element::Type precision = ov::element::dynamic;
    size_t group_size = 0;
    bool by_channel = false;
};

inline bool operator==(const CacheSpec& lhs, const CacheSpec& rhs) {
    return lhs.alg == rhs.alg && lhs.precision == rhs.precision && lhs.group_size == rhs.group_size &&
           lhs.by_channel == rhs.by_channel;
}

/// @brief The quantization group of a head: the whole head unless the configured size divides it.
inline size_t resolve_group_size(size_t configured, size_t head_size) {
    return (configured == 0 || head_size % configured != 0) ? head_size : configured;
}

/// @brief Whether a cache of this precision is stored compressed to u8 / u4 rows (anything else is a plain cache).
inline bool is_quantized_cache(ov::element::Type precision) {
    return precision == ov::element::u8 || precision == ov::element::u4;
}

inline bool is_quantized_cache(const CacheSpec& spec) {
    return is_quantized_cache(spec.precision);
}

}  // namespace ov::intel_cpu
