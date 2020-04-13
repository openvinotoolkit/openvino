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

#ifndef CPU_JIT_UNI_SOFTMAX_KERNEL_F32_HPP
#define CPU_JIT_UNI_SOFTMAX_KERNEL_F32_HPP

#include <cfloat>

#include "c_types_map.hpp"
#include "jit_generator.hpp"
#include "type_helpers.hpp"

#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

template <cpu_isa_t isa>
struct jit_uni_fork_softmax_kernel_f32 : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_softmax_kernel_f32)
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
            isa == avx2, Ymm, Zmm>::type;

    jit_uni_fork_softmax_kernel_f32(jit_softmax_conf_t ajpp) : jpp(ajpp) {
        if (jpp.inner_size > 1)
            this->generate();
        else
            this->generate_dense();

        jit_ker = (decltype(jit_ker))this->getCode();
    }

    jit_softmax_conf_t jpp;

    static status_t init_conf(jit_softmax_conf_t &jpp,
                       const softmax_desc_t &pd,
                       const memory_desc_wrapper &src_d,
                       const memory_desc_wrapper &dst_d);

    void operator()(jit_softmax_call_s *arg) { jit_ker(arg); }

    void prepare_table();
    void simd_expf(const Vmm &vmm_src);
    void scalar_expf(const Xmm &xmm_src);

    void simd_loop_max(int ur_inner);
    void simd_loop_exp(int ur_inner);
    void simd_loop_div(int ur_inner);

    void scalar_loop_max();
    void scalar_loop_exp();
    void scalar_loop_div();

    void dense_loop(int ou_block);
    void generate_dense();
private:
    void (*jit_ker)(jit_softmax_call_s *);

    const int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float);
    const int vlen = cpu_isa_traits<isa>::vlen;

    Reg64 reg_work_amount   = rax;
    Reg64 reg_src_base_ptr  = rbx;
    Reg64 reg_dst_base_ptr  = rsi;
    Reg64 reg_src_ptr       = r8;
    Reg64 reg_dst_ptr       = r9;
    Reg64 reg_channels      = r12;
    Reg64 reg_ch_work       = r13;
    Reg64 reg_min           = rdx;
    Reg64 imm_addr64        = r14;

    Vmm vmm_aux0            = Vmm(0);
    Vmm vmm_aux1            = Vmm(1);
    Vmm vmm_aux2            = Vmm(2);
    Xmm xmm_aux0            = Xmm(0);
    Xmm xmm_aux1            = Xmm(1);
    Xmm xmm_aux2            = Xmm(2);

    Xmm xmm_float_min       = Xmm(3);
    Xmm xmm_one             = Xmm(4);
    Vmm vmm_one             = Vmm(4);

    Xmm xmm_max             = Xmm(5);
    Xmm xmm_denom           = Xmm(6);
    Xmm xmm_src             = Xmm(7);

    Opmask k_mask_tmp       = Opmask(2);

    unsigned char _cmp_gt_os = isa == avx512_common ? 14 : 6;

    int id_vreg_max(int ur_inner);
    int id_vreg_denom(int ur_inner);
    int id_vreg_src(int ur_inner);

    auto vreg_max(int ur_inner) -> Vmm;
    auto vreg_denom(int ur_inner) -> Vmm;
    auto vreg_src(int ur_inner) -> Vmm;

    Label loop_simd_unroll;
    Label loop_simd;
    Label loop_scalar;
    Label loop_end;
    Label l_table;

    unsigned char _op_floor = 1;

    void generate();
};

}
}
}

#endif
