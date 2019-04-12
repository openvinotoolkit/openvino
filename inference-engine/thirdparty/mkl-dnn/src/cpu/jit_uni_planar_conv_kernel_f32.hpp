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

#ifndef JIT_UNI_PLANAR_CONV_KERNEL_F32_HPP
#define JIT_UNI_PLANAR_CONV_KERNEL_F32_HPP

#include "c_types_map.hpp"
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_uni_eltwise.hpp"
#include "jit_uni_depthwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
struct jit_uni_planar_conv_fwd_kernel_f32: public jit_generator {
    jit_uni_planar_conv_fwd_kernel_f32(jit_conv_conf_t ajcp,
            const primitive_attr_t &attr): jcp(ajcp), attr_(attr)
    {
        this->generate();
        jit_ker = (void (*)(jit_conv_call_s *))this->getCode();
    }

    ~jit_uni_planar_conv_fwd_kernel_f32() {
        for (auto inj : eltwise_injectors)
           delete inj;
        eltwise_injectors.clear();

        for (auto inj : depthwise_injectors)
            delete inj;
        depthwise_injectors.clear();
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_planar_conv_fwd_kernel_f32)

    static bool post_ops_ok(jit_conv_conf_t &jcp,
            const primitive_attr_t &attr);
    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d,
            const primitive_attr_t &attr);

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xbyak::Xmm,
            isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    const Xbyak::AddressFrame &vmmword = (isa == sse42)
        ? xword : (isa == avx2) ? yword : zword;

    reg64_t reg_input = r8;
    reg64_t reg_kernel = r9;
    reg64_t reg_output = r10;

    reg64_t aux_reg_input_h = r11;
    reg64_t aux_reg_kernel_h = r12;

    reg64_t aux_reg_input_w = r13;
    reg64_t aux_reg_kernel_w = r14;

    reg64_t aux_reg_inp_d = r9;
    reg64_t aux_reg_ker_d = r10;

    reg64_t reg_kd = rbx;
    reg64_t reg_kh = rdx;
    reg64_t reg_kw = rsi;

    reg64_t kh_iter = rax;
    reg64_t kw_iter = abi_not_param1;

    reg64_t reg_bias = r13;
    reg64_t reg_long_offt = r15;
    reg32_t reg_ci_flag = r15d;

    reg64_t reg_d_weights = r15;
    reg64_t reg_d_bias = kh_iter;

    reg64_t reg_ow = rbp;

    reg64_t reg_oh_blocks = aux_reg_kernel_w;

    reg64_t reg_wj = aux_reg_input_w;

    Vmm vmm_ker = Vmm(15);
    Vmm vmm_tmp = Vmm(15);
    Vmm vmm_src = Vmm(14);
    Xbyak::Xmm xmm_ker = Xbyak::Xmm(15);
    Xbyak::Xmm xmm_tmp = Xbyak::Xmm(15);
    Xbyak::Xmm xmm_src = Xbyak::Xmm(14);

    nstl::vector<jit_uni_eltwise_injector_f32<isa>*> eltwise_injectors;
    nstl::vector<jit_uni_depthwise_injector_f32<isa>*> depthwise_injectors;

    inline void load_src(int ur_h, int ur_w);
    inline void filter(int ur_h);
    inline void filter_unrolled(int ur_h, int ur_w);
    inline void apply_filter(int ur_h, int ur_w);
    inline void apply_postprocess(int ur_h, int ur_w);
    inline void store_dst(int ur_h, int ur_w);
    inline void solve_common(int ur_h);

    inline void filter_scalar(int ur_h);
    inline void load_src_scalar(int ur_h);
    inline void apply_filter_scalar(int ur_h);
    inline void apply_postprocess_scalar(int ur_h);
    inline void store_dst_scalar(int ur_h);

    void generate();
};

}
}
}

#endif
