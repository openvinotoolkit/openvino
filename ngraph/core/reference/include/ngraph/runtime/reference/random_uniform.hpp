// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void random_uniform(const char* out_shape,
                      char* out,
                      const Shape& out_shape_shape,
                      size_t elem_size,
                      int64_t seed,
                      int64_t seed2)
            {

            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
