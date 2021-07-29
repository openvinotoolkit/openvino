// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cfenv>
#include <cmath>
#include <numeric>
#include <vector>

#include "ngraph/runtime/opt_kernel/reshape.hpp"
#include "ngraph/runtime/reference/transpose.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void transpose(const char* data,
                           char* out,
                           const Shape& data_shape,
                           size_t element_size,
                           const int64_t* axes_order,
                           Shape out_shape)
            {
                // To reuse opt_kernel::reshape axes order vector has to be converted to AxisVector
                // Negative axes are not supported, it is validated by transpose evaluate method
                std::vector<size_t> axis_vector(axes_order, axes_order + data_shape.size());
                runtime::opt_kernel::reshape(
                    data, out, data_shape, axis_vector, out_shape, element_size);
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
