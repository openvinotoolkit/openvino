// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace reference {

std::pair<uint64_t, uint64_t> random_uniform(const uint64_t* out_shape,
                                             const char* min_val,
                                             const char* max_val,
                                             char* out,
                                             const Shape& out_shape_shape,
                                             const element::Type& elem_type,
                                             uint64_t seed,
                                             uint64_t seed2,
                                             std::pair<uint64_t, uint64_t> prev_state,
                                             ov::op::PhiloxAlignment alignment = ov::op::PhiloxAlignment::TENSORFLOW);

}  // namespace reference
}  // namespace ov
