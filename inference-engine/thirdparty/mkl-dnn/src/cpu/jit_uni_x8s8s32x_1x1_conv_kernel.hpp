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

#ifndef JIT_UNI_X8S8S32X_1x1_CONV_KERNEL_HPP
#define JIT_UNI_X8S8S32X_1x1_CONV_KERNEL_HPP

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using Xbyak::Reg64;
using Xbyak::Ymm;
using Xbyak::Xmm;

template <cpu_isa_t isa>
struct jit_uni_x8s8s32x_1x1_conv_fwd_kernel: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_x8s8s32x_1x1_conv_fwd_kernel)

    jit_uni_x8s8s32x_1x1_conv_fwd_kernel(jit_1x1_conv_conf_t ajcp,
        const primitive_attr_t &attr): jcp(ajcp), attr_(attr)
    {
        this->generate();
        jit_ker = (void (*)(jit_1x1_conv_call_s *))this->getCode();
    }

    static bool post_ops_ok(jit_1x1_conv_conf_t &jcp,
                            const primitive_attr_t &attr);
    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
                              const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
                              const memory_desc_wrapper &weights_d,
                              const memory_desc_wrapper &dst_d,
                              const memory_desc_wrapper &bias_pd,
                              const primitive_attr_t &attr,
                              bool with_relu = false, float relu_negative_slope = 0.f);

    jit_1x1_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_1x1_conv_call_s *);

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xbyak::Xmm,
            isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;

    Reg64 reg_weight_data = rsi;
    Reg64 reg_src_data = abi_not_param1;
    Reg64 reg_dst_data = rbx;
    Reg64 reg_bias_data = r12;

    Reg64 reg_scales = rdx;
    Reg64 aux_reg_src_data = rdx;
    Reg64 aux_reg_weight_data = rax;
    Reg64 aux_reg_dst_data = rbp;
    Reg64 reg_oc_loop_work = r9;
    Reg64 reg_ow_loop_work = r10;
    Reg64 reg_loop_os_iter = r14;
    Reg64 reg_loop_ic_iter = r15;

    Reg64 reg_scratch = r14;

    Vmm vreg_sum_0 = Vmm(15);
    Vmm vreg_src = Vmm(14);
    Vmm vmm_bias = Vmm(15);
    Vmm vmm_zero = Vmm(14);
    Vmm vmm_one = Vmm(13);
    Xmm xmm_one = Xmm(13);

    void loop_os(int oc_loop_blk);
    void ic_loop(int oc_loop_blk, int ur);

    void generate();

    bool maybe_relu(int position);
    void cvt2ps(data_type_t type_in, Vmm vmm_in, const Xbyak::Operand &op);
};

}
}
}

#endif
