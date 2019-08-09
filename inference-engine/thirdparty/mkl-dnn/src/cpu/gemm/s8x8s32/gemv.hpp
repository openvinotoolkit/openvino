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

#ifndef GEMV_HPP
#define GEMV_HPP

#include <cstdint>

#include "../gemm_info.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <typename T>
int gemm_s8u8s32_jump_to_gemv_s8u8s32(T *arg);
int gemv_threading_driver(gemm_info_t<int8_t, uint8_t, int32_t> *arg);

}
}
}

#endif // GEMV_HPP
