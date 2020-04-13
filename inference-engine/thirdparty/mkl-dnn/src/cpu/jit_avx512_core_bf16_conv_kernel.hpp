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

#ifndef JIT_AVX512_BF16_FWD_KERNEL_HPP
#define JIT_AVX512_BF16_FWD_KERNEL_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"

#include "cpu_memory.hpp"
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_uni_eltwise.hpp"
#include "jit_avx512_core_bf16cvt.hpp"

//#define BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
//#define BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_avx512_core_bf16_fwd_kernel : public jit_generator {

    jit_avx512_core_bf16_fwd_kernel(jit_conv_conf_t ajcp,
            const primitive_attr_t &attr) :
        jit_generator(nullptr, ker_code_size),
        jcp(ajcp),
        attr_(attr),
        eltwise_injector_(nullptr),
        bf16_emu_(nullptr)
    {
        if (jcp.with_eltwise)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx512_common>(
                    this, jcp.eltwise);
        if (!mayiuse(avx512_core_bf16))
            bf16_emu_ = new bf16_emulation_t(this,
                    bf16_emu_reserv_1, bf16_emu_reserv_2,
                    bf16_emu_reserv_3, bf16_emu_reserv_4,
                    bf16_emu_reserv_5, bf16_emu_reserv_6);

        generate();
        jit_ker = (void (*)(jit_conv_call_s *))getCode();
    }

    ~jit_avx512_core_bf16_fwd_kernel() {
        delete bf16_emu_;
        delete eltwise_injector_;
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_bf16_fwd_kernel)

    static bool post_ops_ok(jit_conv_conf_t &jcp,
            const primitive_attr_t &attr);
    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd,
            cpu_memory_t::pd_t &src_pd,
            cpu_memory_t::pd_t &weights_pd,
            cpu_memory_t::pd_t &dst_pd,
            cpu_memory_t::pd_t &bias_pd,
            const primitive_attr_t &attr,
            int nthreads);

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;
    enum {
        ker_reg_base_idx = 28,
        ker_code_size = 1024 * 1024,
    };

    reg64_t param = abi_param1; //L: RDI, W: RCX

    reg64_t reg_inp = r8;
    reg64_t reg_ker = r9;
    reg64_t reg_out = r10;
    reg64_t reg_owb = r11;

    reg64_t aux_reg_inp = r12;
    reg64_t aux_reg_ker = r13;

    reg64_t aux_reg_ker_d = r14;
    reg64_t aux_reg_inp_d = r15;

    reg64_t reg_icb = rax;
    reg64_t reg_bias = rbx;

    reg64_t reg_kj = abi_not_param1;
    reg64_t reg_ki = reg_bias;
    reg64_t reg_oi = rdx;
    reg64_t reg_kh = rsi;

    reg64_t reg_out_long_offt = r14;

    Xbyak::Zmm zmm_out(int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return Xbyak::Zmm(idx);
    }

    Xbyak::Ymm ymm_out(int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return Xbyak::Ymm(idx);
    }

    Xbyak::Zmm zmm_inp(int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return Xbyak::Zmm(idx);
    }

    Xbyak::Ymm ymm_inp(int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return Xbyak::Ymm(idx);
    }
    Xbyak::Reg64 imm_addr64 = r15;
    Xbyak::Zmm zmm_zero = Xbyak::Zmm(31);
    Xbyak::Zmm zmm_wei = Xbyak::Zmm(31);
    Xbyak::Zmm zmm_prev_dst = Xbyak::Zmm(31);
    Xbyak::Zmm zmm_bias = Xbyak::Zmm(31);

    Xbyak::Zmm bf16_emu_reserv_1 = Xbyak::Zmm(26);
    Xbyak::Zmm bf16_emu_reserv_2 = Xbyak::Zmm(27);
    Xbyak::Zmm bf16_emu_reserv_3 = Xbyak::Zmm(28);
    reg64_t bf16_emu_reserv_4 = reg_icb;
    Xbyak::Zmm bf16_emu_reserv_5 = Xbyak::Zmm(29);
    Xbyak::Zmm bf16_emu_reserv_6 = Xbyak::Zmm(30);

    jit_uni_eltwise_injector_f32<avx512_common> *eltwise_injector_;
    bf16_emulation_t *bf16_emu_;

    inline void prepare_output(int ur_w);
    inline void store_output(int ur_w);
    inline void compute_loop(int ur_w, int pad_l, int pad_r);

    void generate();

    size_t get_output_offset(int oi, int n_oc_block) {
        return (size_t)jcp.typesize_out * ((size_t)n_oc_block * jcp.oh
            * jcp.ow * jcp.od + oi) * jcp.oc_block;
    }

    size_t get_input_offset(int ki, int ic, int oi, int pad_l) {
        /*TODO: remove 4vnni */
        size_t scale = (jcp.ver == ver_4vnni || jcp.ver == ver_vnni) ? 2 : 1;
        size_t iw_str = jcp.ic_block;
        size_t ic_str = 1;
        return (size_t)jcp.typesize_in
                * ((size_t)(ki * (jcp.dilate_w + 1) + oi * jcp.stride_w - pad_l)
                          * iw_str
                          + scale * ic * ic_str);
    }

    size_t get_kernel_offset(int ki, int ic,
                                    int n_oc_block, int ker_number) {
        int scale = (jcp.ver == ver_4vnni || jcp.ver == ver_vnni) ? 2 : 1;
        size_t oc_block_stride = (size_t)jcp.nb_ic
                               * jcp.ic_block * jcp.kh * jcp.kw * jcp.kd;
        return jcp.typesize_in * jcp.oc_block
            * (n_oc_block * oc_block_stride
                    + (ic + ker_number) * scale + ki * jcp.ic_block);
    }

    int get_ow_start(int ki, int pad_l) {
        return nstl::max(0,
                utils::div_up(pad_l - ki * (jcp.dilate_w + 1), jcp.stride_w));
    }

    int get_ow_end(int ur_w, int ki, int pad_r) {
        return ur_w - nstl::max(0, utils::div_up(pad_r
                                                   - (jcp.kw - 1 - ki)
                                                           * (jcp.dilate_w + 1),
                                           jcp.stride_w));
    }
};

struct jit_avx512_core_bf16_bwd_data_kernel: public jit_generator {

    jit_avx512_core_bf16_bwd_data_kernel(jit_conv_conf_t ajcp):
        jit_generator(nullptr, ker_code_size),
        jcp(ajcp), bf16_emu_(nullptr)
    {
        if (!mayiuse(avx512_core_bf16))
            bf16_emu_ = new bf16_emulation_t(this,
                    bf16_emu_reserv_1, bf16_emu_reserv_2,
                    bf16_emu_reserv_3, bf16_emu_reserv_4,
                    bf16_emu_reserv_5, bf16_emu_reserv_6);
        generate();
        jit_ker = (void (*)(jit_conv_call_s *))getCode();
    }

    ~jit_avx512_core_bf16_bwd_data_kernel() {
        delete bf16_emu_;
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_bf16_bwd_data_kernel_f32)

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd,
            const memory_desc_wrapper &diff_src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &diff_dst_d);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;
    enum {
        ker_reg_base_idx = 31,
        ker_code_size = 1024 * 1024,
    };

    reg64_t param = abi_param1;
    reg64_t reg_dst = r8;
    reg64_t reg_ker = r9;
    reg64_t reg_src = r10;

    reg64_t aux_reg_dst = r14;
    reg64_t aux_reg_ker = r15;

    reg64_t aux_reg_dst_d = r12;
    reg64_t aux_reg_ker_d = r13;
    reg64_t reg_ki = rsi;

    reg64_t reg_kj = rax;
    reg64_t reg_oi = rbx;
    reg64_t reg_kh = abi_not_param1;

    reg64_t reg_ocb = r11;

    Xbyak::Zmm zmm_inp(int i_ic) {
        int idx = i_ic + jcp.nb_ic_blocking * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return Xbyak::Zmm(idx);
    }

    Xbyak::Ymm ymm_inp(int i_ic) {
        int idx = i_ic + jcp.nb_ic_blocking * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return Xbyak::Ymm(idx);
    }

    Xbyak::Zmm zmm_out(int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return Xbyak::Zmm(idx);
    }

    Xbyak::Zmm bf16_emu_reserv_1 = Xbyak::Zmm(26);
    Xbyak::Zmm bf16_emu_reserv_2 = Xbyak::Zmm(27);
    Xbyak::Zmm bf16_emu_reserv_3 = Xbyak::Zmm(28);
    reg64_t bf16_emu_reserv_4 = reg_kj;
    Xbyak::Zmm bf16_emu_reserv_5 = Xbyak::Zmm(29);
    Xbyak::Zmm bf16_emu_reserv_6 = Xbyak::Zmm(30);

    Xbyak::Zmm zmm_wei = Xbyak::Zmm(31);
    bf16_emulation_t *bf16_emu_;

    inline void prepare_output(int ur_w);
    inline void store_output(int ur_w);
    inline void compute_loop(int ur_w, int l_overflow, int r_overflow);
    void generate();

    int get_iw_start(int ki, int l_overflow)
    {
        int res = (jcp.iw - 1 + jcp.r_pad) % jcp.stride_w
                + l_overflow * jcp.stride_w
                - (jcp.kw - 1 - ki) * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;

        return res;
    }

    int get_iw_end(int ur_w, int ki, int r_overflow)
    {
        if (utils::one_of(ur_w, jcp.iw, jcp.ur_w_tail))
            ur_w += nstl::min(0, jcp.r_pad); // remove negative padding
        int res = (ur_w - 1 + jcp.l_pad) % jcp.stride_w
                + r_overflow * jcp.stride_w - ki * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;

        return ur_w - res;
    }
};

struct jit_avx512_core_bf16_conv_bwd_weights_kernel_f32 : public jit_generator {

    jit_avx512_core_bf16_conv_bwd_weights_kernel_f32(jit_conv_conf_t ajcp) :
        jit_generator(nullptr, ker_code_size),
        jcp(ajcp), bf16_emu_(nullptr)
    {
        if (!isa_has_bf16(jcp.isa)) {
            bf16_emu_ = new bf16_emulation_t(this, one, even, selector, scratch,
                                        tmp0, tmp1);
        }
        generate();
        jit_ker = (void (*)(jit_conv_call_s *))getCode();
    }

    ~jit_avx512_core_bf16_conv_bwd_weights_kernel_f32() {
        delete bf16_emu_;
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_bf16_conv_bwd_weights_kernel_f32)

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, cpu_memory_t::pd_t &src_pd,
            cpu_memory_t::pd_t &diff_weights_pd,
            cpu_memory_t::pd_t &diff_bias_pd, cpu_memory_t::pd_t &diff_dst_pd);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_s *);

private:
    Xbyak::Label dst_prm_table;

    using reg64_t = const Xbyak::Reg64;
    enum {
        typesize = sizeof(float),
        ker_code_size = 1024 * 1024,
    };
    static const int max_ur_w;

    reg64_t param = abi_param1;
    reg64_t reg_input = rax;
    reg64_t reg_kernel = rdx;
    reg64_t reg_output = rsi;
    reg64_t b_ic = abi_not_param1;
    reg64_t kj = r8;
    reg64_t reg_kh = r9;
    reg64_t reg_ur_w_trips = r10;
    reg64_t reg_oj = r15;
    reg64_t reg_ih_count = rbx;
    reg64_t reg_tmp = r14;
    reg64_t reg_long_offt = r14;

    reg64_t ki = r11;
    reg64_t reg_kd_count = r12;
    reg64_t reg_oi = r12;
    reg64_t reg_d_index = r13;
    reg64_t reg_input_d = r15;
    reg64_t reg_output_d = rbx;
    reg64_t aux_reg_input = r12;
    reg64_t aux_reg_kernel = r13;
    reg64_t reg_bias = rbx;

    Xbyak::Zmm one = Xbyak::Zmm(27);
    Xbyak::Zmm even = Xbyak::Zmm(28);
    Xbyak::Zmm selector = Xbyak::Zmm(29);
    Xbyak::Zmm tmp0 = Xbyak::Zmm(30);
    Xbyak::Zmm tmp1 = Xbyak::Zmm(31);
    reg64_t scratch = r11;

    inline void maybe_zero_kernel();
    inline void compute_oh_step_unroll_ow_icblock(int ic_block_step,
            int max_ur_w);
    inline void od_step_comeback_pointers();
    inline void oh_step_comeback_pointers();
    inline void compute_oh_step_unroll_ow(int ic_block_step, int max_ur_w);
    inline void compute_ic_block_step(int ur_w,
            int pad_l, int pad_r, int ic_block_step,
            int input_offset, int kernel_offset, int output_offset,
            bool is_tail = false);
    inline void compute_oh_step_common(int ic_block_step, int max_ur_w);
    inline void compute_oh_step_disp();
    inline void compute_loop();
    inline void compute_oh_loop_common();
    inline void compute_od_loop_common();

    void generate();

    static void balance(const jit_conv_conf_t &j, int &nthr, int &nthr_mb,
            int &nthr_g, int &nthr_oc_b, int &nthr_ic_b);

    bf16_emulation_t *bf16_emu_;
#ifdef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    int stack_space_needed = 64;
#endif
};

}
}
}
#endif
