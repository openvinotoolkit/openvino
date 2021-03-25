// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "ngraph/axis_vector.hpp"
#include "ngraph/check.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/op/pad.hpp" // for op::PadMode

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void pad(const char* data,
                     const char* pad_value,
                     char* out,
                     const size_t elem_size,
                     const Shape& data_shape,
                     const Shape& out_shape,
                     const CoordinateDiff& padding_below,
                     const CoordinateDiff& padding_above,
                     const op::PadMode pad_mode);
        }
    }
}
