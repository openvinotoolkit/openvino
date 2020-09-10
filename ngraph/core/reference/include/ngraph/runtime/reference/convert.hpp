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

#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO
#include <tbb/parallel_for.h>
#endif

#include "ngraph/type/float16.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO
            template <typename TI, typename TO>
            void convert(const TI* arg, TO* out, size_t count)
            {
                tbb::parallel_for(tbb::blocked_range<size_t>(0, count),
                                  [&](tbb::blocked_range<size_t> range) {
                                      for (size_t i = range.begin(); i < range.end(); ++i)
                                      {
                                          out[i] = static_cast<TO>(arg[i]);
                                      }
                                  });
            }

            template <>
            void convert<uint8_t, float16>(const uint8_t* arg, float16* out, size_t count);

            template <typename T>
            void convert_to_bool(const T* arg, char* out, size_t count)
            {
                tbb::parallel_for(tbb::blocked_range<size_t>(0, count),
                                  [&](tbb::blocked_range<size_t> range) {
                                      for (size_t i = range.begin(); i < range.end(); ++i)
                                      {
                                          out[i] = static_cast<char>(static_cast<bool>(arg[i]));
                                      }
                                  });
            }
#else
            template <typename TI, typename TO>
            void convert(const TI* arg, TO* out, size_t count)
            {
                for (size_t i = 0; i < count; ++i)
                {
                    out[i] = static_cast<TO>(arg[i]);
                }
            }

            template <typename T>
            void convert_to_bool(const T* arg, char* out, size_t count)
            {
                for (size_t i = 0; i < count; ++i)
                {
                    out[i] = static_cast<char>(static_cast<bool>(arg[i]));
                }
            }
#endif
        }
    }
}
