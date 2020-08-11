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

#include <cstddef>

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void relu(const T* arg, T* out, size_t count)
            {
                T zero = 0;
                for (size_t i = 0; i < count; i++)
                {
                    out[i] = arg[i] > zero ? arg[i] : zero;
                }
            }
            template <typename T>
            void relu_backprop(const T* arg, const T* delta_arg, T* out, size_t count)
            {
                T zero = 0;
                for (size_t i = 0; i < count; i++)
                {
                    out[i] = arg[i] > zero ? delta_arg[i] : zero;
                }
            }
        }
    }
}
