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

#ifndef JIT_SSE41_GEMV_T_F32_KERN_HPP
#define JIT_SSE41_GEMV_T_F32_KERN_HPP

#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

class jit_sse42_gemv_t_f32_kern : public jit_generator {
public:
    jit_sse42_gemv_t_f32_kern(void);
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse42_gemv_t_f32_kern);

protected:
    void v_load(const Xbyak::Xmm &dst, const Xbyak::Address &src, int nelems);
    void v_store(const Xbyak::Address &dst, const Xbyak::Xmm &src, int nelems);

    void dot_product(const Xbyak::Xmm &dst, const Xbyak::Xmm &src1,
            const Xbyak::Xmm &src2);
    void innerloop(int unroll_m, int unroll_n);
    void outerloop(int unroll_x, int unroll_y, Xbyak::Label *&outerloop_label);

    void generate();

private:
    static const int M_UNROLL_ = 8;
    static const int N_UNROLL_ = 4;

    static const int size_ = 4;

    static const int offset_a_ = 128, offset_x_ = 128;

    // Integer register assignments
    Xbyak::Reg64 M_, N_, A_, LDA_, X_, INCY_, Y_, ALPHA_, I_, J_;
    Xbyak::Reg64 AO_, XO_, YO_, YO2_;

    // Vector register assignments
    Xbyak::Xmm scratch_, alpha_, a_regs_[M_UNROLL_ >> 2][N_UNROLL_];
    Xbyak::Xmm x_regs_[M_UNROLL_ >> 2], y_regs_[N_UNROLL_], acc_[N_UNROLL_];

    // Stack variable assignments
    Xbyak::Address arg_lda_, arg_x_, arg_incy_, arg_y_;
};

} // namespace cpu
} // namespace impl
} // namespace mkldnn

#endif // JIT_SSE41_GEMV_T_F32_KERN_HPP
