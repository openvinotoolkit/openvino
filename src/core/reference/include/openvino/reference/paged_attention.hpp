// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/paged_attention.hpp"

namespace ov {
namespace reference {

std::pair<uint64_t, uint64_t> paged_attention(const uint64_t* out_shape,
                                             const char* min_val,
                                             const char* max_val,
                                             char* out,
                                             const Shape& out_shape_shape,
                                             const element::Type& elem_type,
                                             uint64_t seed,
                                             uint64_t seed2,
                                             std::pair<uint64_t, uint64_t> prev_state);

}  // namespace reference
}  // namespace ov
