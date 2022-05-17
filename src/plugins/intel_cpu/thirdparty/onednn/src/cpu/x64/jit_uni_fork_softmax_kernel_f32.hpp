/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef CPU_X64_JIT_UNI_FORK_SOFTMAX_KERNEL_F32_HPP
#define CPU_X64_JIT_UNI_FORK_SOFTMAX_KERNEL_F32_HPP

#include <cfloat>
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

template <cpu_isa_t isa>
struct jit_uni_fork_softmax_kernel_f32 : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_softmax_kernel_f32)
    using Vmm = typename utils::conditional3<isa == sse41, Xmm,
            isa == avx2, Ymm, Zmm>::type;

    jit_uni_fork_softmax_kernel_f32(jit_softmax_conf_t ajpp);

    jit_softmax_conf_t jpp;

    static status_t init_conf(jit_softmax_conf_t &jpp,
                       const softmax_desc_t &pd,
                       const memory_desc_wrapper &src_d,
                       const memory_desc_wrapper &dst_d);

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
    Reg64 bf16_emu_gpr      = r15;

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

    Zmm bf16_emu_zmm_1      = Zmm(27);
    Zmm bf16_emu_zmm_2      = Zmm(28);
    Zmm bf16_emu_zmm_3      = Zmm(29);
    Zmm bf16_emu_zmm_4      = Zmm(30);
    Zmm bf16_emu_zmm_5      = Zmm(31);

    Opmask k_mask_tmp       = Opmask(2);

    unsigned char _cmp_gt_os = isa == avx512_common ? 14 : 6;

    int id_vreg_max(int ur_inner);
    int id_vreg_denom(int ur_inner);
    int id_vreg_src(int ur_inner);

    auto vreg_max(int ur_inner) -> Vmm;
    auto vreg_denom(int ur_inner) -> Vmm;
    auto vreg_src(int ur_inner) -> Vmm;

    void load_vector(Vmm vmm_src, const Xbyak::Address &op);
    void load_scalar(Xmm xmm_src, const Xbyak::Address &op);
    void store_vector(const Xbyak::Address &op, Vmm vmm_dst);
    void store_scalar(const Xbyak::Address &op, Xmm xmm_dst);

    Label loop_simd_unroll;
    Label loop_simd;
    Label loop_scalar;
    Label loop_end;
    Label l_table;

    std::unique_ptr<bf16_emulation_t> bf16_emu_;

    unsigned char _op_floor = 1;

    void generate() override;
};

}
}
}
}

#endif
