//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <cmath>
#include <cstring>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T, typename U>
            void embedding(const U* indices,
                           const T* weights,
                           T* out,
                           size_t indices_count,
                           const Shape& out_shape)
            {
                size_t vec_len = out_shape.at(1);
                T* out_iter = out;
                for (size_t i = 0; i < indices_count; i++)
                {
                    memcpy(out_iter,
                           &weights[vec_len * static_cast<size_t>(indices[i])],
                           sizeof(T) * vec_len);
                    out_iter += vec_len;
                }
            }
        }
    }
}
