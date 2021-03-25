// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cfenv>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T, typename U>
            void transpose(const T* arg, T* out, Shape arg_size, const U* axes_order = nullptr)
            {
                std::vector<size_t> range_vector;
                if (axes_order == nullptr)
                {
                    range_vector.resize(arg_size.size());
                    std::iota(range_vector.begin(), range_vector.end(), 0);
                    std::reverse(range_vector.begin(), range_vector.end());
                    axes_order = range_vector.data();
                }
                size_t cnt = 0;
                for (size_t i = 0; i < arg_size.size(); ++i)
                {
                    size_t axes = axes_order[i];
                    size_t start = 0;
                    for (size_t j = 0; j < axes; ++j)
                    {
                        start += shape_size(arg_size[j]);
                    }
                    for (size_t j = start; j < start + shape_size(arg_size[axes]); ++j)
                    {
                        out[cnt++] = arg[j];
                    }
                }
            }
        }
    }
}
