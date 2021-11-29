// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void reorg_yolo(const char* arg,
                            char* out,
                            const Shape& in_shape,
                            int64_t stride,
                            const size_t elem_size);
        }
    } // namespace runtime
} // namespace ngraph
