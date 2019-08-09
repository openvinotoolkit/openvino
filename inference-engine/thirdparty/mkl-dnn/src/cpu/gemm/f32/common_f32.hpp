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

#ifndef COMMON_F32_HPP
#define COMMON_F32_HPP

#include "jit_generator.hpp"

#define F32_COPY_KERNEL_CODE_SIZE          (4096L * 5)

namespace mkldnn {
namespace impl {
namespace cpu {

class jit_avx512_core_f32_copy_an_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_f32_copy_an_kern);

    public:
        jit_avx512_core_f32_copy_an_kern();
};

class jit_avx512_core_f32_copy_at_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_f32_copy_at_kern);

    public:
        jit_avx512_core_f32_copy_at_kern();
};

class jit_avx512_core_f32_copy_bn_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_f32_copy_bn_kern);

    public:
        jit_avx512_core_f32_copy_bn_kern();
};

class jit_avx512_core_f32_copy_bt_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_f32_copy_bt_kern);

    public:
        jit_avx512_core_f32_copy_bt_kern();
};

class jit_avx2_f32_copy_an_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_f32_copy_an_kern);

    public:
        jit_avx2_f32_copy_an_kern();
};

class jit_avx2_f32_copy_at_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_f32_copy_at_kern);

    public:
        jit_avx2_f32_copy_at_kern();
};

class jit_avx2_f32_copy_bn_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_f32_copy_bn_kern);

    public:
        jit_avx2_f32_copy_bn_kern();
};

class jit_avx2_f32_copy_bt_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_f32_copy_bt_kern);

    public:
        jit_avx2_f32_copy_bt_kern();
};

}
}
}
#endif // COMMON_F32_HPP
