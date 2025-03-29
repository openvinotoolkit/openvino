// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"
#include "openvino/op/space_to_depth.hpp"

namespace ov {
namespace reference {
void space_to_depth(const char* const in,
                    const Shape& in_shape,
                    char* const out,
                    const Shape& out_shape,
                    const size_t block_size,
                    const op::v0::SpaceToDepth::SpaceToDepthMode mode,
                    const size_t elem_size);
}
}  // namespace ov
