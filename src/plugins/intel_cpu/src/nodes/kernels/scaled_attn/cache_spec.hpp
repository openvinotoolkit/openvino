// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/internal_properties.hpp"

namespace ov::Extensions::Cpu {

struct CacheSpec {
    ov::internal::CacheQuantAlgorithm alg = ov::internal::CacheQuantAlgorithm::SCALAR;
    ov::element::Type precision = ov::element::dynamic;
    size_t group_size = 0;
    bool by_channel = false;
};

/// @brief The quantization group of a head: the whole head unless the configured size divides it.
inline size_t resolve_group_size(size_t configured, size_t head_size) {
    return (configured == 0 || head_size % configured != 0) ? head_size : configured;
}

/// @brief Whether the cache is stored compressed to u8 / u4 rows (anything else is a plain cache).
inline bool is_quantized_cache(const CacheSpec& spec) {
    return spec.precision == ov::element::u8 || spec.precision == ov::element::u4;
}

}  // namespace ov::Extensions::Cpu
