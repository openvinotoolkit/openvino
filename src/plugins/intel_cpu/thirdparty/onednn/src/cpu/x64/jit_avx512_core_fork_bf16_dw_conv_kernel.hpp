/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef CPU_X64_JIT_AVX512_CORE_FORK_BF16_DW_CONV_KERNEL_HPP
#define CPU_X64_JIT_AVX512_CORE_FORK_BF16_DW_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"

#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/injectors/jit_uni_depthwise_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_avx512_fork_dw_conv_fwd_kernel_bf16 : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_fork_dw_conv_fwd_kernel_bf16)

    jit_avx512_fork_dw_conv_fwd_kernel_bf16(const jit_conv_conf_t &ajcp, const memory_desc_t &dst_md, const primitive_attr_t& attr)
        : jcp(ajcp), attr_(attr), bf16_emu_(nullptr) {
        if (!isa_has_bf16(jcp.isa))
            bf16_emu_ = new bf16_emulation_t(this, bf16_emu_reserv_1,
                    bf16_emu_reserv_2, bf16_emu_reserv_3, bf16_emu_reserv_4,
                    bf16_emu_reserv_5, bf16_emu_reserv_6);
    }

    ~jit_avx512_fork_dw_conv_fwd_kernel_bf16() {
        for (auto inj : eltwise_injectors)
            delete inj;
        eltwise_injectors.clear();

        for (auto inj : depthwise_injectors)
            delete inj;
        depthwise_injectors.clear();

        delete bf16_emu_;
    }

    jit_conv_conf_t jcp;
    const primitive_attr_t& attr_;

private:
    using reg64_t = const Xbyak::Reg64;
    using mask_t = const Xbyak::Opmask;
    const Xbyak::AddressFrame &vmmword = zword;

    const int acc_idx_start = 2;
    inline int get_max_regs() { return isa_has_bf16(jcp.isa) ? 30 : 25; };

    // dw convolution
    reg64_t reg_input = r8;
    reg64_t aux_reg_input = r9;
    reg64_t aux1_reg_input = r10;
    reg64_t reg_kernel = r11;
    reg64_t aux_reg_kernel = r12;
    reg64_t reg_ch_blocks = r13;
    reg64_t reg_output = r14;
    reg64_t reg_bias = r15;
    reg64_t reg_kh = rax;
    reg64_t reg_kw = rbx;
    reg64_t iter_kh = rdx;
    reg64_t iter_kw = rsi;
    reg64_t reg_ur_w = rbp;
    reg64_t reg_tail = abi_not_param1;
    reg64_t aux1_reg_kernel = reg_ch_blocks;
    reg64_t imm_addr64 = aux1_reg_input;
    reg64_t reg_d_weights = imm_addr64;
    reg64_t reg_d_bias = iter_kh;
    int base_post_ops_data_offset = 0;
    constexpr static int reg64_size = 8;
    reg64_t aux_reg_ch_blocks = reg_ur_w;
    reg64_t aux_reg_blocks_offset = reg_tail;

    mask_t k_oc_tail_mask = Xbyak::Opmask(2);
    mask_t ktail_mask = k_oc_tail_mask;
    mask_t k_ch_tail_mask_extended = Xbyak::Opmask(3);

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

    inline bool is_src_layout_nxc() {
        return utils::one_of(jcp.src_tag, format_tag::ndhwc, format_tag::nhwc,
                             format_tag::nwc);
    }

    inline bool is_dst_layout_nxc() {
        return utils::one_of(jcp.dst_tag, format_tag::ndhwc, format_tag::nhwc,
                             format_tag::nwc);
    }

    inline void load_src(int ur_ch_blocks, int ur_w, bool last_ch_block_flag);
    inline void compute_loop(int ur_w, int ur_ch_blocks);
    inline void apply_filter(int ur_ch_blocks, int ur_w, bool last_ch_block_flag);
    inline void apply_filter_unrolled(int ur_ch_blocks, int ur_w, bool last_ch_block_flag);
    inline void apply_postprocess(int ur_ch_blocks, int ur_w);
    inline void store_dst(int ur_ch_blocks, int ur_w, bool last_ch_block_flag);
    inline void loop_ow(int ur_ch_blocks);

    nstl::vector<jit_uni_eltwise_injector_f32<avx512_common>*> eltwise_injectors;
    nstl::vector<jit_uni_depthwise_injector_f32<avx512_common>*> depthwise_injectors;

    bf16_emulation_t *bf16_emu_;

    void generate() override;
};

struct jit_avx512_fork_dw_conv_bwd_data_kernel_bf16 : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_fork_dw_conv_bwd_data_kernel_bf16)

    jit_avx512_fork_dw_conv_bwd_data_kernel_bf16(const jit_conv_conf_t &ajcp, const primitive_attr_t&)
        : jcp(ajcp), bf16_emu_(nullptr) {

        if (!isa_has_bf16(jcp.isa))
            bf16_emu_ = new bf16_emulation_t(this, bf16_emu_reserv_1,
                    bf16_emu_reserv_2, bf16_emu_reserv_3, bf16_emu_reserv_4,
                    bf16_emu_reserv_5, bf16_emu_reserv_6);
    }

    ~jit_avx512_fork_dw_conv_bwd_data_kernel_bf16() { delete bf16_emu_; }

    jit_conv_conf_t jcp;

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

    void generate() override;
};

}
}
}
}

#endif
