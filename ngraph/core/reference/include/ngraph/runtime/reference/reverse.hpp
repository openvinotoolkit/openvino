// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void reverse(const char* arg,
                         char* out,
                         const Shape& arg_shape,
                         const Shape& out_shape,
                         const AxisSet& reversed_axes,
                         size_t elem_size);
        }
    } // namespace runtime
} // namespace ngraph
