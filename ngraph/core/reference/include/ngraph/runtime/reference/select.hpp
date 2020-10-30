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
#include <iostream>

#include "ngraph/runtime/reference/autobroadcast_binop.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void select(const char* arg0,
                        const T* arg1,
                        const T* arg2,
                        T* out,
                        size_t arg0_count,
                        size_t arg1_count,
                        size_t arg2_count,
                        size_t out_count)
            {
                for (size_t i = 0; i < out_count; i++)
                {
                    out[i] = arg0[i % arg0_count] ? arg1[i % arg1_count] : arg2[i % arg2_count];
                }
            }

            template <typename T>
            void select(const char* arg0,
                        const T* arg1,
                        const T* arg2,
                        T* out,
                        const Shape& arg0_shape,
                        const Shape& arg1_shape,
                        const Shape& arg2_shape,
                        const op::AutoBroadcastSpec& broadcast_spec)
            {
                autobroadcast_select(
                    arg0,
                    arg1,
                    arg2,
                    out,
                    arg0_shape,
                    arg1_shape,
                    arg2_shape,
                    broadcast_spec,
                    [](char s, T x, T y) -> T { return static_cast<T>(s ? x : y); });
            }
        }
    }
}
