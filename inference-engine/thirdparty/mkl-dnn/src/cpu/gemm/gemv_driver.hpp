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

#ifndef GEMV_DRIVER_HPP
#define GEMV_DRIVER_HPP

#include "mkldnn_types.h"
#include "gemm_info.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <typename a_t, typename b_t, typename c_t>
mkldnn_status_t jump_to_gemv(const gemm_info_t<a_t, b_t, c_t> *arg);
}
} // namespace impl
} // namespace mkldnn

#endif // GEMV_DRIVER_HPP
