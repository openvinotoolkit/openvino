// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"
#include "openvino/op/depth_to_space.hpp"

namespace ov {
namespace reference {
void depth_to_space(const char* const in,
                    const Shape& in_shape,
                    char* const out,
                    const Shape& out_shape,
                    const size_t block_size,
                    const op::v0::DepthToSpace::DepthToSpaceMode mode,
                    const size_t elem_size);
}
}  // namespace ov
