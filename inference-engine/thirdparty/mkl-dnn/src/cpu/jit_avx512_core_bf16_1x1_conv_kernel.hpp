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

#ifndef JIT_AVX512_CORE_BF16_1x1_CONV_KERNEL_HPP
#define JIT_AVX512_CORE_BF16_1x1_CONV_KERNEL_HPP

#include "c_types_map.hpp"
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_uni_eltwise.hpp"
#include "jit_avx512_core_bf16cvt.hpp"

//#define BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION

namespace mkldnn {
namespace impl {
namespace cpu {

namespace { const size_t code_size_bf16_bwd_w = 1024 * 1024; }

struct jit_avx512_core_bf16_1x1_conv_kernel : public jit_generator {
    jit_avx512_core_bf16_1x1_conv_kernel(jit_1x1_conv_conf_t ajcp,
            const primitive_attr_t &attr) :
    jit_generator(nullptr, ker_code_size),
    jcp(ajcp), attr_(attr)
    , eltwise_injector_(nullptr)
    , bf16_emu_(nullptr)
    {
        if (jcp.with_eltwise)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx512_common>(
                    this, jcp.eltwise);

        if (!mayiuse(avx512_core_bf16))
            bf16_emu_ = new bf16_emulation_t(this,
                    bf16_emu_reserv_1, bf16_emu_reserv_2,
                    bf16_emu_reserv_3, bf16_emu_reserv_4,
                    bf16_emu_reserv_5, bf16_emu_reserv_6);

        this->generate();
        jit_ker = (void (*)(jit_1x1_conv_call_s *)) this->getCode();
    }

    ~jit_avx512_core_bf16_1x1_conv_kernel() {
        delete eltwise_injector_;
        delete bf16_emu_;
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_bf16_1x1_conv_kernel)

    static bool post_ops_ok(jit_1x1_conv_conf_t &jcp,
                                const primitive_attr_t &attr);

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
            const convolution_desc_t &cd,
            const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d,
            const memory_desc_wrapper &bias_d,
            const primitive_attr_t &attr,
            int nthreads, bool reduce_src);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_1x1_conv_conf_t &jcp);

    jit_1x1_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_1x1_conv_call_s *);

  private:
    using reg64_t = const Xbyak::Reg64;
    using zmm_t = const Xbyak::Zmm;
    using mask_t = const Xbyak::Opmask;
    enum {
        ker_code_size = 1024 * 1024,
    };

    reg64_t reg_bcast_data = r8;
    reg64_t reg_load_data = r10;
    reg64_t reg_output_data = r9;
    reg64_t aux_reg_bcast_data = r14;
    reg64_t aux1_reg_bcast_data = rbx;
    reg64_t aux_reg_load_data = r15;
    reg64_t imm_addr64 = aux_reg_load_data;
    reg64_t aux_reg_output_data = abi_not_param1;
    reg64_t reg_load_loop_work = rsi;
    reg64_t reg_reduce_loop_work = r11;
    reg64_t bcast_loop_iter = rdx;
    reg64_t reduce_loop_iter = abi_param1;
    reg64_t reg_reduce_pos_flag = rax;
    reg64_t reg_output_stride = r13;
    reg64_t reg_bias_data = r12;
    reg64_t reg_bcast_loop_work = aux1_reg_bcast_data;
    reg64_t reg_trans_tmp = rax;

    mask_t vmask = k7;

    Xbyak::Xmm xmm_relu_ns = Xbyak::Xmm(30);
    Xbyak::Zmm zmm_relu_ns = Xbyak::Zmm(30);
    Xbyak::Zmm zmm_zero = Xbyak::Zmm(31);
    Xbyak::Zmm vreg_bcast = Xbyak::Zmm(31);

    Xbyak::Zmm bf16_emu_reserv_1 = Xbyak::Zmm(25);
    Xbyak::Zmm bf16_emu_reserv_2 = Xbyak::Zmm(26);
    Xbyak::Zmm bf16_emu_reserv_3 = Xbyak::Zmm(27);
    reg64_t bf16_emu_reserv_4 = imm_addr64;
    Xbyak::Zmm bf16_emu_reserv_5 = Xbyak::Zmm(28);
    Xbyak::Zmm bf16_emu_reserv_6 = Xbyak::Zmm(29);

    Xbyak::Zmm zmm_tmp2 = Xbyak::Zmm(30);
    Xbyak::Zmm zmm_bias = Xbyak::Zmm(31);

    Xbyak::Label dst_prm_table;

    jit_uni_eltwise_injector_f32<avx512_common> *eltwise_injector_;

    int bcast_loop_work_offt = 0;
#ifdef BF16_CONV_1x1_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    int perm_reg_offset = 8;
    int broadcast_space = 24;
#endif
    int stack_space_needed = 96;

    void bcast_loop(int load_loop_blk);
    void reduce_loop(int load_loop_blk, int ur, int substep, bool wraparound);

    void generate();
    static void balance(jit_1x1_conv_conf_t &jcp, int nthreads);

    bf16_emulation_t *bf16_emu_;
};
}
}
}

#endif
