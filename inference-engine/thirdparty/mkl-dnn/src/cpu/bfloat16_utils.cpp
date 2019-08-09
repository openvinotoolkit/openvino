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

#include "bfloat16_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {
namespace bf16_cvt_utils {

jit_avx512_core_cvt_ps_to_bf16_t cvt_one_ps_to_bf16(1);
jit_avx512_core_cvt_ps_to_bf16_t cvt_ps_to_bf16_;
jit_avx512_core_cvt_bf16_to_ps_t cvt_bf16_to_ps_;
jit_avx512_core_add_cvt_ps_to_bf16_t add_cvt_ps_to_bf16_;

}
}
}
}
