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

#ifndef BFLOAT16_UTILS_HPP
#define BFLOAT16_UTILS_HPP

#include "nstl.hpp"
#include "jit_avx512_core_bf16cvt.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {
namespace bf16_cvt_utils {

union f32_bf16_t {
    float vfloat;
    mkldnn_bfloat16_t vbfloat[2];
};

jit_avx512_core_cvt_ps_to_bf16_t &cvt_one_ps_to_bf16();
jit_avx512_core_cvt_ps_to_bf16_t &cvt_ps_to_bf16_();
jit_avx512_core_cvt_bf16_to_ps_t &cvt_bf16_to_ps_();
jit_avx512_core_add_cvt_ps_to_bf16_t &add_cvt_ps_to_bf16_();

inline mkldnn_bfloat16_t cvt_float_to_bfloat16(float inp) {
    assert(mayiuse(avx512_core));
    mkldnn_bfloat16_t out;
    jit_call_t p;
    p.inp = (void *)&inp;
    p.out = (void *)&out;
    cvt_one_ps_to_bf16().jit_ker(&p);
    return out;
}

inline void cvt_float_to_bfloat16(mkldnn_bfloat16_t *out, const float *inp) {
    assert(mayiuse(avx512_core));
    jit_call_t p;
    p.inp = (void *)inp;
    p.out = (void *)out;
    cvt_one_ps_to_bf16().jit_ker(&p);
}

inline void cvt_float_to_bfloat16(mkldnn_bfloat16_t *out, const float *inp,
        size_t size) {
    assert(mayiuse(avx512_core));
    jit_call_t p_;
    p_.inp = (void *)inp;
    p_.out = (void *)out;
    p_.size = size;
    cvt_ps_to_bf16_().jit_ker(&p_);
 }

inline float cvt_bfloat16_to_float(mkldnn_bfloat16_t inp) {
    assert(mayiuse(avx512_core));
    f32_bf16_t cvt = {0};
    cvt.vbfloat[1] = inp;
    return cvt.vfloat;
}

inline void cvt_bfloat16_to_float(float *out, const mkldnn_bfloat16_t *inp) {
    assert(mayiuse(avx512_core));
    f32_bf16_t cvt = {0};
    cvt.vbfloat[1] = *inp;
    *out = cvt.vfloat;
}

inline void cvt_bfloat16_to_float(float *out, const mkldnn_bfloat16_t *inp,
        size_t size) {
    assert(mayiuse(avx512_core));
    jit_call_t p_;
    p_.inp = (void *)inp;
    p_.out = (void *)out;
    p_.size = size;
    cvt_bf16_to_ps_().jit_ker(&p_);
}

// performs element-by-element sum of inp and add float arrays and stores
// result to bfloat16 out array with downconversion
inline void add_floats_and_cvt_to_bfloat16(mkldnn_bfloat16_t *out,
        const float *inp0,
        const float *inp1,
        size_t size) {
    assert(mayiuse(avx512_core));
    jit_call_t p_;
    p_.inp = (void *)inp0;
    p_.add = (void *)inp1;
    p_.out = (void *)out;
    p_.size = size;
    add_cvt_ps_to_bf16_().jit_ker(&p_);
}

inline mkldnn_bfloat16_t approx_bfloat16_lowest() {
    /* jit fails to convert FLT_MIN to bfloat16.
     * It converst FLT_MIN to -INF. Truncate FLT_MIN
     * to bfloat16 to get a value close to minimum bfloat16*/
    f32_bf16_t f_raw = {0};
    f_raw.vfloat = nstl::numeric_limits<float>::lowest();
    f_raw.vbfloat[0] = 0;
    return f_raw.vbfloat[1];
}

inline bool is_float_representable_in_bfloat16(float x) {
    f32_bf16_t cvt = {0};
    cvt.vfloat = x;
    return cvt.vbfloat[0] == 0;
}

}
}
}
}

#endif
