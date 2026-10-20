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

}  // namespace ov::Extensions::Cpu
