// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/space_to_depth.hpp"
#include "ngraph/shape.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
void space_to_depth(const char* const in,
                    const Shape& in_shape,
                    char* const out,
                    const Shape& out_shape,
                    const size_t block_size,
                    const op::SpaceToDepth::SpaceToDepthMode mode,
                    const size_t elem_size);
}
}  // namespace runtime
}  // namespace ngraph
