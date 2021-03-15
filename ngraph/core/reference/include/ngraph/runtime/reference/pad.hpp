//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/util/attr_types.hpp" // for op::PadMode
#include "ngraph/shape.hpp"

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
    } // namespace runtime
} // namespace ngraph
