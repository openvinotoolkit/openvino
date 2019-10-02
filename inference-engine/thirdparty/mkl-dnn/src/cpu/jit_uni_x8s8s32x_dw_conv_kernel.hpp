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

#ifndef JIT_UNI_X8S8S32X_DW_CONV_KERNEL_F32_HPP
#define JIT_UNI_X8S8S32X_DW_CONV_KERNEL_F32_HPP

#include "c_types_map.hpp"
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"
#include "type_helpers.hpp"
#include "jit_uni_eltwise.hpp"
#include "jit_uni_depthwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
struct jit_uni_x8s8s32x_dw_conv_fwd_kernel: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_dw_conv_fwd_kernel_f32)

    jit_uni_x8s8s32x_dw_conv_fwd_kernel(jit_conv_conf_t ajcp,
            const primitive_attr_t &attr): jcp(ajcp), attr_(attr) {
        this->generate();
        jit_ker = (void (*)(jit_conv_call_s *))this->getCode();
    }

    ~jit_uni_x8s8s32x_dw_conv_fwd_kernel() {
        for (auto inj : eltwise_injectors)
           delete inj;
        eltwise_injectors.clear();

        for (auto inj : depthwise_injectors)
            delete inj;
        depthwise_injectors.clear();
    }

    static bool post_ops_ok(jit_conv_conf_t &jcp,
            const primitive_attr_t &attr);
    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d,
            const memory_desc_wrapper &bias_pd,
            const primitive_attr_t &attr);

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xbyak::Xmm,
        isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using Ymm = const Xbyak::Ymm;
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using reg8_t = const Xbyak::Reg8;
    const int vlen = cpu_isa_traits<isa>::vlen;

    reg64_t aux_reg_inp_d = r8;
    reg64_t aux_reg_ker_d = r12;
    reg64_t reg_kd = abi_not_param1;

    reg64_t reg_input_base = r10;
    reg64_t reg_output_base = r9;
    reg64_t reg_kernel_base = r11;
    reg64_t reg_ch_work = r13;
    reg64_t reg_bias_base = abi_not_param1;
    reg64_t reg_scales_base = rdx;

    reg64_t reg_input = r8;
    reg64_t reg_kernel = r12;
    reg64_t aux_reg_input = r9;
    reg64_t aux1_reg_input = r10;
    reg64_t aux_reg_kernel = r13;
    reg64_t aux1_reg_kernel = r11;
    reg64_t reg_output = r14;

    reg64_t reg_kh = rax;
    reg64_t reg_kw = rbx;
    reg64_t iter_kh = rdx;
    reg64_t iter_kw = rsi;
    reg64_t reg_ur_w = rbp;

    reg32_t reg_tmp_32 = r15d;
    reg64_t reg_tmp_64 = r15;
    reg8_t reg_tmp_8 = r15b;

    reg64_t imm_addr64 = r10;

    reg64_t reg_oc_off = iter_kw;
    reg64_t reg_d_weights = aux1_reg_kernel;
    reg64_t reg_d_bias = aux_reg_input;

    Vmm vmm_zero = Vmm(0);
    Vmm vmm_bias = Vmm(3);
    Vmm vmm_scale = Vmm(2);
    Vmm vmm_prev_dst = Vmm(2);

    inline Vmm get_ker_reg(int idx) { return Vmm(idx + 0); }
    inline Vmm get_src_reg(int idx) { return Vmm(idx + 1); }
    inline Vmm get_acc_reg(int idx) { return Vmm(idx + 4); }

    inline void cvt2ps(data_type_t type_in, Vmm vmm_in, const Xbyak::Operand &op, bool scalar_load);
    inline void store_dst(const Xbyak::Address &op, Vmm vmm_dst, bool scalar_store);

    inline void load_src(int ur_ch_blocks, int ch_step, int ur_w);
    inline void apply_filter(int ur_ch_blocks, int ch_step, int ur_w);
    inline void apply_filter_unrolled(int ur_ch_blocks, int ch_step, int ur_w);
    inline void store_dst(int ur_ch_blocks, int ch_step, int ur_w);
    inline void loop_body(int ur_ch_blocks, int ch_step);

    inline void prepare_table();
    void generate();

    nstl::vector<jit_uni_eltwise_injector_f32<isa>*> eltwise_injectors;
    nstl::vector<jit_uni_depthwise_injector_f32<isa>*> depthwise_injectors;

    Xbyak::Label l_table;
};

}
}
}

#endif
