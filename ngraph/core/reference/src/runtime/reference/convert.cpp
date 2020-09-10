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

#include <cstring>

#include "ngraph/runtime/reference/convert.hpp"

#include <immintrin.h>

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO
            template <>
            void convert<uint8_t, float16>(const uint8_t* arg, float16* out, size_t count)
            {
                size_t const n = count / 8;

                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, n), [&](tbb::blocked_range<size_t> range) {
                        for (size_t i = range.begin(); i < range.end(); ++i)
                        {
                            // TODO: Check CPU flags

                            __m128i u8vec =
                                _mm_loadl_epi64((__m128i const*)&arg[i * 8]); // SSE2: Load(u8[8])
                            __m256i i32vec =
                                _mm256_cvtepu8_epi32(u8vec); // AVX2: Convert u8[8] -> i32[8]
                            __m256 fvec =
                                _mm256_cvtepi32_ps(i32vec); // AVX: Convert i32[8] -> float[8]
                            __m128i f16vec =
                                _mm256_cvtps_ph(fvec, 0); // FP16C: Convert float[8] -> float16[8]
                            _mm_storeu_si128(
                                (__m128i*)&out[i * 8],
                                f16vec); // SSE2: Store (Works only in LE architecture!)
                        }
                    });

                for (size_t i = n * 8; i < count; ++i)
                {
                    out[i] = static_cast<float16>(arg[i]);
                }
            }
#endif
        }
    }
}
