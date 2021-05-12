// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void tile(const char* arg,
                      char* out,
                      const Shape& in_shape,
                      const Shape& out_shape,
                      const size_t elem_size,
                      const std::vector<int64_t>& repeats);
        }
    } // namespace runtime
} // namespace ngraph
