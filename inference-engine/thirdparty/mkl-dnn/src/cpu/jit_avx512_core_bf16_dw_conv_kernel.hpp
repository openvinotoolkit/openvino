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

#ifndef JIT_AVX512_DW_CONV_KERNEL_BF16_HPP
#define JIT_AVX512_DW_CONV_KERNEL_BF16_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"

#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_uni_eltwise.hpp"

#include "jit_avx512_core_bf16cvt.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_avx512_dw_conv_fwd_kernel_bf16 : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_dw_conv_fwd_kernel_bf16)

    jit_avx512_dw_conv_fwd_kernel_bf16(jit_conv_conf_t ajcp, const primitive_attr_t&)
        : jcp(ajcp), eltwise_injector_(nullptr), bf16_emu_(nullptr) {
        if (jcp.with_eltwise)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx512_common>(
                    this, jcp.eltwise);
        if (!mayiuse(avx512_core_bf16))
            bf16_emu_ = new bf16_emulation_t(this, bf16_emu_reserv_1,
                    bf16_emu_reserv_2, bf16_emu_reserv_3, bf16_emu_reserv_4,
                    bf16_emu_reserv_5, bf16_emu_reserv_6);

        this->generate();
        jit_ker = (void (*)(jit_conv_call_s *)) this->getCode();
    }

    ~jit_avx512_dw_conv_fwd_kernel_bf16() {
        delete eltwise_injector_;
        delete bf16_emu_;
    }

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;
    const Xbyak::AddressFrame &vmmword = zword;

    const int acc_idx_start = 2;
    inline int get_max_regs() { return isa_has_bf16(jcp.isa) ? 30 : 25; };

    // dw convolution
    reg64_t reg_input = r8;
    reg64_t aux_reg_input = r9;
    reg64_t aux1_reg_input = r10;
    reg64_t reg_kernel = r11;
    reg64_t aux_reg_kernel = r12;
    reg64_t aux1_reg_kernel = r13;
    reg64_t reg_output = r14;
    reg64_t reg_bias = r15;
    reg64_t reg_kh = rax;
    reg64_t reg_kw = rbx;
    reg64_t iter_kh = rdx;
    reg64_t iter_kw = rsi;
    reg64_t reg_ur_w = rbp;
    reg64_t reg_ch_blocks = aux1_reg_input;
    reg64_t imm_addr64 = aux1_reg_input;

    Xbyak::Zmm zmm_ker_reg = Xbyak::Zmm(0);
    Xbyak::Zmm zmm_src_reg = Xbyak::Zmm(1);
    Xbyak::Zmm zmm_prev_dst = Xbyak::Zmm(31);

    /* Registers used for bfloat16 emulation */
    Xbyak::Zmm bf16_emu_reserv_1 = Xbyak::Zmm(26);
    Xbyak::Zmm bf16_emu_reserv_2 = Xbyak::Zmm(27);
    Xbyak::Zmm bf16_emu_reserv_3 = Xbyak::Zmm(28);
    reg64_t bf16_emu_reserv_4 = iter_kw;
    Xbyak::Zmm bf16_emu_reserv_5 = Xbyak::Zmm(29);
    Xbyak::Zmm bf16_emu_reserv_6 = Xbyak::Zmm(30);

    inline Xbyak::Zmm get_acc_reg(int idx) {
        assert(idx + acc_idx_start <= get_max_regs());
        return Xbyak::Zmm(idx + acc_idx_start);
    }

    inline void load_src(int ur_ch_blocks, int ur_w);
    inline void apply_filter(int ur_ch_blocks, int ur_w);
    inline void apply_filter_unrolled(int ur_ch_blocks, int ur_w);
    inline void apply_activation(int ur_ch_blocks, int ur_w);
    inline void store_dst(int ur_ch_blocks, int ur_w);
    inline void loop_ow(int ur_ch_blocks);

    jit_uni_eltwise_injector_f32<avx512_common> *eltwise_injector_;

    bf16_emulation_t *bf16_emu_;

    void generate();
};

struct jit_avx512_dw_conv_bwd_data_kernel_bf16 : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_dw_conv_bwd_data_kernel_bf16)

    jit_avx512_dw_conv_bwd_data_kernel_bf16(jit_conv_conf_t ajcp, const primitive_attr_t&)
        : jcp(ajcp), bf16_emu_(nullptr) {

        if (!mayiuse(avx512_core_bf16))
            bf16_emu_ = new bf16_emulation_t(this, bf16_emu_reserv_1,
                    bf16_emu_reserv_2, bf16_emu_reserv_3, bf16_emu_reserv_4,
                    bf16_emu_reserv_5, bf16_emu_reserv_6);

        this->generate();
        jit_ker = (void (*)(jit_conv_call_s *))this->getCode();
    }

    ~jit_avx512_dw_conv_bwd_data_kernel_bf16() {
        delete bf16_emu_;
    }

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;

    const int acc_idx_start = 2;
    inline int get_max_regs() { return isa_has_bf16(jcp.isa) ? 30 : 25; };

    Xbyak::Zmm zmm_ker_reg = Xbyak::Zmm(0);
    Xbyak::Zmm zmm_dst_reg = Xbyak::Zmm(1);

    inline Xbyak::Zmm get_acc_reg(int idx) {
        assert(idx + acc_idx_start <= get_max_regs());
        return Xbyak::Zmm(idx + acc_idx_start);
    }

    reg64_t reg_ddst = rax;
    reg64_t aux_reg_ddst = r8;
    reg64_t aux1_reg_ddst = abi_not_param1;
    reg64_t reg_kernel = rdx;
    reg64_t aux_reg_kernel = r10;
    reg64_t aux1_reg_kernel = rbp;
    reg64_t reg_dsrc = rsi;

    reg64_t reg_ur_str_w = r9;
    reg64_t reg_ch_blocks = rbx;

    reg64_t iter_kh = r11;
    reg64_t iter_kw = r12;
    reg64_t reg_kh = r13;
    reg64_t reg_kw = r14;

    Xbyak::Zmm bf16_emu_reserv_1 = Xbyak::Zmm(26);
    Xbyak::Zmm bf16_emu_reserv_2 = Xbyak::Zmm(27);
    Xbyak::Zmm bf16_emu_reserv_3 = Xbyak::Zmm(28);
    reg64_t bf16_emu_reserv_4 = iter_kw;
    Xbyak::Zmm bf16_emu_reserv_5 = Xbyak::Zmm(29);
    Xbyak::Zmm bf16_emu_reserv_6 = Xbyak::Zmm(30);

    bf16_emulation_t *bf16_emu_;

    inline void loop_body(int ur_ch_blocks);
    inline void load_ddst(int ur_ch_blocks, int ur_str_w);
    inline void apply_filter(int ur_ch_blocks, int ur_str_w);
    inline void store_dsrc(int ur_ch_blocks, int ur_str_w);

    void generate();
};

struct jit_avx512_dw_conv_bwd_weights_kernel_bf16 : public jit_generator {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_dw_conv_bwd_weights_kernel_bf16)

    jit_avx512_dw_conv_bwd_weights_kernel_bf16(jit_conv_conf_t ajcp)
        : jcp(ajcp), bf16_emu_(nullptr) {

        if (!mayiuse(avx512_core_bf16))
            bf16_emu_ = new bf16_emulation_t(this, bf16_emu_reserv_1,
                    bf16_emu_reserv_2, bf16_emu_reserv_3, bf16_emu_reserv_4,
                    bf16_emu_reserv_5, bf16_emu_reserv_6);

        this->generate();
        jit_ker = (void (*)(jit_dw_conv_call_s *)) this->getCode();
    }

    ~jit_avx512_dw_conv_bwd_weights_kernel_bf16() { delete bf16_emu_; }

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_dw_conv_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;
    const Xbyak::AddressFrame &vmmword = zword;

    const int idx_start = 2;
    inline int get_max_regs() { return isa_has_bf16(jcp.isa) ? 30 : 25; };

    /* Offset between input and accummulators is 3, therefore, assume 'kw'
     * is no larger than 3*/
    Xbyak::Zmm zmm_bias_reg = Xbyak::Zmm(0);
    Xbyak::Zmm zmm_out_reg = Xbyak::Zmm(1);

    inline Xbyak::Zmm get_acc_reg(int idx) {
        assert(idx + idx_start <= get_max_regs());
        return Xbyak::Zmm(idx + idx_start);
    }
    inline Xbyak::Zmm get_input_reg(int idx) {
        const int i_idx = idx_start + jcp.kw + idx % jcp.kw;
        assert(i_idx <= get_max_regs());
        return Xbyak::Zmm(i_idx);
    }

    reg64_t reg_tmp_input = r9;
    reg64_t reg_tmp_output = r10;
    reg64_t reg_tmp_filter = r13;
    reg64_t reg_kh_offset = rax;

    /* parameter passed by driver into kernel */
    Xbyak::Reg8 reg_exec_flags = bl;
    reg64_t reg_oh_worksize = r14;
    reg64_t reg_oh = rax;
    reg64_t iter_ow_blk = r11;
    reg64_t reg_kh = rsi;
    reg64_t reg_kh_count = rdx;

    /* Base addresses for convolution parameters. */
    reg64_t reg_input_baddr = r15;
    reg64_t reg_output_baddr = r12;
    reg64_t reg_filter_baddr = abi_not_param1;
    reg64_t reg_bias_baddr = r13;

    /* Registers used for bfloat16 emulation */
    Xbyak::Zmm bf16_emu_reserv_1 = Xbyak::Zmm(26);
    Xbyak::Zmm bf16_emu_reserv_2 = Xbyak::Zmm(27);
    Xbyak::Zmm bf16_emu_reserv_3 = Xbyak::Zmm(28);
    reg64_t bf16_emu_reserv_4 = r8;
    Xbyak::Zmm bf16_emu_reserv_5 = Xbyak::Zmm(29);
    Xbyak::Zmm bf16_emu_reserv_6 = Xbyak::Zmm(30);

    bf16_emulation_t *bf16_emu_;

    /* Micro-kernel JIT'ing, fusing 'kw' and 'ow_block' loops into unrolled FMAs
     */
    inline void compute_ow_step_unroll(
            int unroll_w, int l_pad, int pad_offset, int ow_block);

    /* JIT'ing the outer loops for the micro-kernel -> {kh, oh_block} */
    inline void compute_h_step(
            int unroll_w, int l_pad, int pad_offset, int ow_block);
    inline void compute_h_loop(
            int unroll_w, int l_pad, int pad_offset, int ow_block);

    /* Write 'width' micro-kernel JITs; depending on the padding and convolution
     * size, write a micro-kernel for the left ow-block, middle ow-block(s), and
     * right ow-block.*/
    inline void compute_ow_block_unroll();

    inline void compute_zero_filter();
    inline void load_filter();
    inline void zero_filter();
    inline void load_bias();
    inline void zero_bias();
    inline void compute_bias_step_unroll(const int unroll_w);
    inline void compute_bias_loop(const int block_size);
    inline void store_filter();
    inline void store_bias();

    void generate();
};

}
}
}

#endif /* JIT_UNI_DW_CONV_KERNEL_BF16_HPP */
