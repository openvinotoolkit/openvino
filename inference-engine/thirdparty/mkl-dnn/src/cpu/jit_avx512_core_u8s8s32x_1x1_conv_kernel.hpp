/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#ifndef JIT_AVX512_CORE_U8S8S32X_1X1_CONV_KERNEL_HPP
#define JIT_AVX512_CORE_U8S8S32X_1X1_CONV_KERNEL_HPP

#include "c_types_map.hpp"
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_avx512_core_u8s8s32x_1x1_conv_kernel: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_u8s8s32x_1x1_conv_fwd_ker_t)
    jit_avx512_core_u8s8s32x_1x1_conv_kernel(jit_1x1_conv_conf_t ajcp,
            const primitive_attr_t &attr) : jcp(ajcp), attr_(attr)
    {
        this->generate();
        jit_ker = (void (*)(jit_1x1_conv_call_s *)) this->getCode();
    }

    static bool post_ops_ok(jit_1x1_conv_conf_t &jcp,
                                const primitive_attr_t &attr);

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
                                const convolution_desc_t &cd,
                                const memory_desc_wrapper &src_d,
                                const memory_desc_wrapper &weights_d,
                                const memory_desc_wrapper &dst_d,
                                const memory_desc_wrapper &bias_d,
                                const primitive_attr_t &attr,
                                bool with_relu, float relu_negative_slope,
                                int nthreads, bool reduce_src);

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
                              const convolution_desc_t &cd,
                              const memory_desc_wrapper &src_d,
                              const memory_desc_wrapper &weights_d,
                              const memory_desc_wrapper &dst_d,
                              const memory_desc_wrapper &bias_d,
                              const primitive_attr_t &attr,
                              int nthreads, bool reduce_src)
    {
        return init_conf(jcp, cd, src_d, weights_d, dst_d, bias_d, attr, false,
            0.0, nthreads, reduce_src);
    }
    bool maybe_relu(int position);

    jit_1x1_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_1x1_conv_call_s *);

  private:
    using reg64_t = const Xbyak::Reg64;
    using zmm_t = const Xbyak::Zmm;
    using mask_t = const Xbyak::Opmask;

    reg64_t reg_bcast_data = r8;
    reg64_t reg_ptr_scales = r8;
    reg64_t reg_output_data = r9;
    reg64_t reg_load_data = r10;
    reg64_t reg_ptr_sum_scale = r10;
    reg64_t reg_reduce_loop_work = r11;
    reg64_t reg_bias_data = r12;
    reg64_t aux_reg_acc_s32 = r12;
    reg64_t reg_acc_s32 = r13;
    reg64_t reg_scratch = r13;
    reg64_t aux_reg_bcast_data = r14;
    reg64_t aux_reg_load_data = r15;
    reg64_t imm_addr64 = r15;
    reg64_t reg_reduce_pos_flag = rax;
    reg64_t aux1_reg_bcast_data = rbx;
    reg64_t reg_bcast_loop_work = rbx;
    reg64_t bcast_loop_iter = rdx; // Note: Fix me
    reg64_t reg_load_loop_work = rsi;
    reg64_t aux_reg_output_data = abi_not_param1;
    reg64_t reduce_loop_iter = abi_param1;

    mask_t vmask = k7;

    Xbyak::Zmm zmm_tmp = Xbyak::Zmm(28);
    Xbyak::Zmm zmm_one = Xbyak::Zmm(29);
    Xbyak::Zmm zmm_zero = Xbyak::Zmm(30);
    Xbyak::Zmm zmm_bcast = Xbyak::Zmm(31);

    int bcast_loop_work_offt = 0;
    int reg_bias_data_offt = 8;
    int aux_reg_acc_s32_offt = 16;
    int reg_bcast_data_off = 24;
    int reg_load_data_off = 32;
    int reg_ptr_sum_scale_off = 40;
    int stack_space_needed = 48;

    void bcast_loop(int load_loop_blk);
    void reduce_loop(int load_loop_blk, int ur, int substep, bool wraparound);

    void generate();
    static void balance(jit_1x1_conv_conf_t &jcp, int nthreads);
};
}
}
}

#endif
