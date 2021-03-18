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

#include <cmath>
#include <cstddef>
#include <ngraph/op/gelu.hpp>

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void gelu(const T* arg, T* out, op::GeluApproximationMode mode, size_t count)
            {
                if (mode == op::GeluApproximationMode::ERF)
                {
                    for (size_t i = 0; i < count; i++)
                    {
                        out[i] = 0.5 * arg[i] * (1 + erf(arg[i] / std::sqrt(2.0)));
                    }
                }
                else if (mode == op::GeluApproximationMode::TANH)
                {
                    const auto pi = atan(1.0) * 4.0;
                    const auto sqpi = std::sqrt(2.0 / pi);
                    for (size_t i = 0; i < count; i++)
                    {
                        auto& x = arg[i];
                        out[i] =
                            0.5 * x * (1.0 + std::tanh(sqpi * (x + 0.044715 * std::pow(x, 3))));
                    }
                }
            }
        }
    }
}
