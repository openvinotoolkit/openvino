/*******************************************************************************
* Copyright 2018-2019 Intel Corporation
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

#ifndef JIT_UNI_X8S8S32X_CONV_KERNEL_HPP
#define JIT_UNI_X8S8S32X_CONV_KERNEL_HPP

#include "c_types_map.hpp"
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"
#include "cpu_memory.hpp"
#include "jit_uni_eltwise.hpp"
#include "jit_uni_depthwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
struct jit_uni_x8s8s32x_conv_fwd_kernel: public jit_generator {
    jit_uni_x8s8s32x_conv_fwd_kernel(jit_conv_conf_t ajcp, jit_conv_conf_t ajcp_dw,
            const primitive_attr_t &attr): jcp(ajcp), jcp_dw(ajcp_dw), attr_(attr)
    {
        this->generate();
        jit_ker = (void (*)(jit_conv_call_s *))this->getCode();
    }

    ~jit_uni_x8s8s32x_conv_fwd_kernel() {
        for (auto inj : eltwise_injectors)
           delete inj;
        eltwise_injectors.clear();

        for (auto inj : depthwise_injectors)
            delete inj;
        depthwise_injectors.clear();
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_x8s8s32x_conv_fwd_kernel)

    static bool post_ops_ok(jit_conv_conf_t &jcp,
            const primitive_attr_t &attr);
    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd,
            cpu_memory_t::pd_t &src_pd,
            cpu_memory_t::pd_t &weights_pd,
            cpu_memory_t::pd_t &dst_pd,
            cpu_memory_t::pd_t &bias_pd,
            const primitive_attr_t &attr);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp, const jit_conv_conf_t &jcp_dw, const primitive_attr_t &attr);

    jit_conv_conf_t jcp;
    jit_conv_conf_t jcp_dw;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xbyak::Xmm,
            isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    const int vlen = cpu_isa_traits<isa>::vlen;
    using Ymm = const Xbyak::Ymm;
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using reg8_t = const Xbyak::Reg8;

    reg64_t aux_reg_inp_d = r11;
    reg64_t aux_reg_ker_d = abi_not_param1;
    reg64_t reg_kd = rsi;

    reg64_t reg_scales_base = r13;
    reg64_t reg_bias_base = rbp;
    reg64_t reg_input_base = r8;
    reg64_t reg_output_base = r9;
    reg64_t reg_kernel_base = rbx;

    reg64_t reg_input = rax;
    reg64_t aux_reg_input = r8;
    reg64_t aux1_reg_input = r13;
    reg64_t reg_kernel = rdx;
    reg64_t aux_reg_kernel = r9;
    reg64_t aux1_reg_kernel = rbx;
    reg64_t reg_output = rsi;

    reg64_t reg_kj = r10;
    reg64_t reg_overflow = r10;
    reg64_t reg_oi_iter = r11;
    reg64_t reg_ic_iter = r15;
    reg64_t reg_compensation_base = abi_not_param1;
    reg64_t reg_oc_work = r12;
    reg64_t imm_addr64 = rbx;

    reg8_t reg_tmp_8 = r14b;
    reg32_t reg_tmp_32 = r14d;
    reg64_t reg_tmp_64 = r14;

    reg64_t reg_oc_off = r10;
    reg64_t reg_d_weights = aux_reg_kernel;
    reg64_t reg_d_bias = aux_reg_input;

    reg64_t reg_input_zp = r14;
    reg64_t reg_weights_zp_compensation = aux_reg_input;
    reg64_t reg_weights_zp_compensation_base = r14;
    reg64_t reg_table = aux_reg_kernel;
    reg64_t reg_ci_flag = r14;

    Vmm vmm_one = Vmm(15);
    Vmm vmm_bias_alpha = Vmm(13);
    Vmm vmm_shift = Vmm(14);
    Vmm vmm_bias = Vmm(15);
    Ymm ymm_tmp = Ymm(10);
    Vmm vmm_scale = Vmm(12);
    Vmm vmm_comp = Vmm(12);
    Vmm vmm_prev_dst = Vmm(12);

    Vmm vmm_d_weights = Vmm(14);
    Vmm vmm_d_bias = Vmm(15);

    inline Vmm get_src_reg(int idx) { return Vmm(idx + 9); }
    inline Vmm get_ker_reg(int idx) { return Vmm(idx + 0); }
    inline Vmm get_tmp_reg(int idx) { return Vmm(idx + 13); }
    inline Vmm get_acc_reg(int idx) { return Vmm(idx + 1); }

    inline void cvt2ps(data_type_t type_in, Vmm ymm_in, const Xbyak::Operand &op, bool scalar_load);
    inline void store_dst(const Xbyak::Address &op, Vmm vmm_dst, bool scalar_store, bool need_pack = true);

    inline void apply_filter(int ur_w, int pad_l, int pad_r, int oc_blocks, int oc_step,
                             int ic_tail_size, bool h_padded, bool first_oc_block);
    inline void oh_step_unroll_kw(int ur_w, int pad_l, int pad_r, int oc_blocks, int oc_step, bool h_padded, bool first_oc_block);
    inline void kh_loop(int ur_w, int pad_l, int pad_r, int oc_blocks, int oc_step, bool first_oc_block);
    inline void kd_loop(int ur_w, int pad_l, int pad_r, int oc_blocks, int oc_step, bool first_oc_block);
    inline void width_blk_step(int ur_w, int pad_l, int pad_r, int oc_blocks, int oc_step, bool first_oc_block);
    inline void solve_common(int oc_blocks, int oc_step, bool first_oc_block);

    void generate();

    void prepare_table();

    nstl::vector<jit_uni_eltwise_injector_f32<isa>*> eltwise_injectors;
    nstl::vector<jit_uni_depthwise_injector_f32<isa>*> depthwise_injectors;

    Xbyak::Label l_table;
};

}
}
}

#endif
