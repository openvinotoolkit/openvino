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

#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void erf(const T* arg, T* out, size_t count)
            {
                for (size_t i = 0; i < count; i++)
                {
                    out[i] = std::erf(arg[i]);
                }
            }

            template <>
            void erf<int8_t>(const int8_t* arg, int8_t* out, size_t count)
            {
                for (size_t i = 0; i < count; i++)
                {
                    out[i] = std::round(std::erf(arg[i]));
                }
            }

            template <>
            void erf<int16_t>(const int16_t* arg, int16_t* out, size_t count)
            {
                for (size_t i = 0; i < count; i++)
                {
                    out[i] = std::round(std::erf(arg[i]));
                }
            }

            template <>
            void erf<int32_t>(const int32_t* arg, int32_t* out, size_t count)
            {
                for (size_t i = 0; i < count; i++)
                {
                    out[i] = std::round(std::erf(arg[i]));
                }
            }
        }
    }
}
