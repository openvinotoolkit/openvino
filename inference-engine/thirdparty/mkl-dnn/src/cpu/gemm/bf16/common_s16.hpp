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

#ifndef COMMON_S16_HPP
#define COMMON_S16_HPP

#include "jit_generator.hpp"

#define S16_COPY_KERNEL_CODE_SIZE          (4096L * 8)

namespace mkldnn {
namespace impl {
namespace cpu {

class jit_avx512_core_s16_copy_an_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_s16_copy_an_kern);

    public:
        jit_avx512_core_s16_copy_an_kern();
};

class jit_avx512_core_s16_copy_at_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_s16_copy_at_kern);

    public:
        jit_avx512_core_s16_copy_at_kern();
};

class jit_avx512_core_s16_copy_bn_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_s16_copy_bn_kern);

    public:
        jit_avx512_core_s16_copy_bn_kern();
};

class jit_avx512_core_s16_copy_bt_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_s16_copy_bt_kern);

    public:
        jit_avx512_core_s16_copy_bt_kern();
};

}
}
}
#endif // COMMON_S16_HPP
