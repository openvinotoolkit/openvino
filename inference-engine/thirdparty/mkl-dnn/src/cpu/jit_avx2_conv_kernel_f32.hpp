/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#ifndef JIT_AVX2_CONV_KERNEL_F32_HPP
#define JIT_AVX2_CONV_KERNEL_F32_HPP

#include "c_types_map.hpp"
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_uni_eltwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_avx2_conv_fwd_kernel_f32: public jit_generator {
    jit_avx2_conv_fwd_kernel_f32(jit_conv_conf_t ajcp,
            const primitive_attr_t &attr): jcp(ajcp), attr_(attr)
    {
        this->generate();
        jit_ker = (void (*)(jit_conv_call_s *))this->getCode();
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_conv_fwd_kernel_f32)

    static bool post_ops_ok(jit_conv_conf_t &jcp,
            const primitive_attr_t &attr);
    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d,
            const primitive_attr_t &attr,
            bool with_relu = false,
            float relu_negative_slope = 0.);

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;
    reg64_t reg_input = rax;
    reg64_t aux_reg_input = r8;
    reg64_t reg_kernel = rdx;
    reg64_t aux_reg_kernel = r9;
    reg64_t reg_output = rsi;
    reg64_t reg_bias = rbx;

    reg64_t kj = r10;
    reg64_t oi_iter = r11;
    reg64_t ki_iter = r12;
    reg64_t reg_kh = abi_not_param1;
    reg64_t reg_oc_blocks = r14;
    reg64_t imm_addr64 = r15;
    Xbyak::Reg32 reg_ci_flag = r13d;

    Xbyak::Label l_table;
    jit_uni_eltwise_vector_f32<avx2> eltwise_generator;

    inline void oh_step_unroll_kw(int ur_w, int pad_l, int pad_r,
            int oc_blocks);
    inline void oh_step_nopad(int ur_w, int pad_l, int pad_r,
            char pad_label, int oc_blocks, char oc_blocks_label);
    inline void width_blk_step(int ur_w, int pad_l, int pad_r,
            char pad_label, int oc_blocks, char oc_blocks_label);
    inline void solve_common(int oc_blocks, char oc_blocks_label);

    void generate();
};

struct jit_avx2_conv_bwd_data_kernel_f32: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_conv_bwd_data_kernel_f32)

    jit_avx2_conv_bwd_data_kernel_f32(jit_conv_conf_t ajcp): jcp(ajcp)
    {
        this->generate();
        jit_ker = (void (*)(jit_conv_call_s *))this->getCode();
    }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &diff_src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &diff_dst_d);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;

    reg64_t reg_input      = rax;
    reg64_t reg_ddst       = rax;
    reg64_t aux_reg_input  = r8;
    reg64_t aux_reg_ddst   = r8;
    reg64_t aux1_reg_input = r9;
    reg64_t reg_kernel     = rdx;
    reg64_t aux_reg_kernel = r10;
    reg64_t reg_output     = rsi;
    reg64_t reg_dsrc       = rsi;
    reg64_t aux_reg_output = rbx;
    reg64_t aux_reg_dsrc = rbx;

    reg64_t kj      = r11;
    reg64_t oi_iter = r12;
    reg64_t reg_kh  = r14;
    reg64_t ki_iter = r13;
    reg64_t start_off_reg = r15;

    inline void hsw_iter(int ur_w, int l_overflow, int r_overflow,
            int start_off, char hsw_iter_tag, char start_off_tag);

    void generate();
};

struct jit_avx2_conv_bwd_weights_kernel_f32: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_conv_bwd_weights_kernel_f32)

    jit_avx2_conv_bwd_weights_kernel_f32(jit_conv_conf_t ajcp): jcp(ajcp)
    {
        this->generate();
        jit_ker = (void (*)(jit_conv_call_s *))this->getCode();
    }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &diff_weights_d,
            const memory_desc_wrapper &diff_dst_d);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;
    reg64_t reg_input = rax;
    reg64_t reg_kernel = rdx;
    reg64_t reg_output = rsi;
    reg64_t b_ic = abi_not_param1;
    reg64_t kj = r8;
    reg64_t reg_kh = r9;
    reg64_t reg_ur_w_trips = r10;
    reg64_t reg_tmp = r11;
    reg64_t reg_oj = r15;
    reg64_t reg_ih_count = rbx;

    inline void oh_step_comeback_pointers(const char *kh_comeback_label);
    inline void compute_ic_block_step(int ur_w, int pad_l, int pad_r,
            int ic_block_step, int input_offset, int kernel_offset,
            int output_offset);
    inline void compute_oh_step_disp(const char* kh_label,
            const char* ic_block_label, const char* ow_block_label,
            const char* kh_comeback_label);
    inline void compute_oh_step_unroll_ow(const char* kh_label,
            const char* ic_block_label, const char* ow_block_label,
            const char* kh_comeback_label, int ic_block_step, int max_ur_w);
    inline void compute_oh_step_common(const char* kh_label,
            const char* ic_block_label, const char* ow_block_label,
            const char* kh_comeback_label, int ic_block_step, int max_ur_w);
    inline void compute_oh_loop_common();

    void generate();
};

}
}
}

#endif
