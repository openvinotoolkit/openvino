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

#include <cmath>
#include <cstddef>

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void selu(const T* arg,
                      const T* alpha,
                      const T* lambda,
                      T* out,
                      size_t size_arg,
                      size_t size_alpha,
                      size_t size_lambda)
            {
                for (size_t i = 0; i < size_arg; ++i)
                {
                    out[i] = arg[i] > T(0) ? T(lambda[i % size_lambda] * arg[i])
                                           : T(alpha[i % size_alpha] * lambda[i % size_lambda] *
                                               (std::exp(arg[i]) - 1));
                }
            }
        }
    }
}
