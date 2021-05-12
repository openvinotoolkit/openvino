// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void concat(const std::vector<const char*>& args,
                        char* out,
                        const std::vector<Shape>& in_shapes,
                        const Shape& out_shape,
                        int64_t concatenation_axis,
                        size_t elem_size);
        }
    } // namespace runtime
} // namespace ngraph
