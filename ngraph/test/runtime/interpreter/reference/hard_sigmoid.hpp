//*****************************************************************************
// Copyright 2020 Intel Corporation
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
            template <typename T>
            void hard_sigmoid(const T* arg,
                              const T* alpha,
                              const T* beta,
                              T* out,
                              size_t size_arg,
                              size_t size_alpha,
                              size_t size_beta)
            {
                int cnt = 0;
                for (size_t i = 0; i < size_arg; ++i)
                {
                    out[i] = std::max(
                        T(0),
                        std::min(T(1),
                                 T(alpha[cnt % size_alpha] * arg[i] + beta[cnt % size_beta])));
                    cnt++;
                }
            }
        }
    }
}
