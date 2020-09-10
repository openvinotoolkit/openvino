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

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            T round_to_nearest_even(const T arg)
            {
                const auto floor_arg = std::floor(arg);
                const auto diff = arg - floor_arg;
                if (diff < 0.5f || (diff == 0.5f && static_cast<int>(floor_arg) % 2 == 0))
                {
                    return floor_arg;
                }
                else
                {
                    return floor_arg + 1.0f;
                }
            }

            template <typename T>
            void round(const T* arg, T* out, size_t count)
            {
                for (size_t i = 0; i < count; ++i)
                {
                    out[i] = round_to_nearest_even(arg[i]);
                }
            }
        }
    }
}
