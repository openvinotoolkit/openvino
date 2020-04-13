/*******************************************************************************
* Copyright 2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef JIT_AVX512_CORE_GEMV_S8X8S32_HPP
#define JIT_AVX512_CORE_GEMV_S8X8S32_HPP

#include <cstdint>

#include "../gemm_info.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <typename T>
int jump_to_gemv_s8x8s32(T *arg);

} // namespace cpu
} // namespace impl
} // namespace mkldnn

#endif // JIT_AVX512_CORE_GEMV_S8X8S32_HPP
