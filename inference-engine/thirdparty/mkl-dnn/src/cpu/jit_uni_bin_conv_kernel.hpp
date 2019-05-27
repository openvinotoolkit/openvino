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

#ifndef JIT_UNI_BIN_CONV_KERNEL_HPP
#define JIT_UNI_BIN_CONV_KERNEL_HPP

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
struct jit_uni_bin_conv_fwd_kernel: public jit_generator {
    jit_uni_bin_conv_fwd_kernel(jit_bin_conv_conf_t ajcp, jit_conv_conf_t ajcp_dw_conv,
            const primitive_attr_t &attr): jcp(ajcp), jcp_dw_conv(ajcp_dw_conv), attr_(attr)
    {
        this->generate();
        jit_ker = (void (*)(jit_conv_call_s *))this->getCode();
    }

    ~jit_uni_bin_conv_fwd_kernel() {
        for (auto inj : eltwise_injectors)
           delete inj;
        eltwise_injectors.clear();

        for (auto inj : depthwise_injectors)
            delete inj;
        depthwise_injectors.clear();
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_bin_conv_fwd_kernel)

    static bool post_ops_ok(jit_bin_conv_conf_t &jcp, const primitive_attr_t &attr);
    static status_t init_conf(jit_bin_conv_conf_t &jcp,
            const binary_convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d, const primitive_attr_t &attr);
    static void init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_bin_conv_conf_t &jcp, const jit_conv_conf_t &jcp_dw_conv);

    jit_bin_conv_conf_t jcp;
    jit_conv_conf_t jcp_dw_conv;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xbyak::Xmm,
            isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using Ymm = const Xbyak::Ymm;
    using reg8_t = const Xbyak::Reg8;
    using reg16_t = const Xbyak::Reg16;
    using reg32_t = const Xbyak::Reg32;
    using reg64_t = const Xbyak::Reg64;

    reg64_t reg_input = r13;
    reg64_t reg_output = rbp;
    reg64_t reg_input_base = rax;
    reg64_t aux_reg_input = r8;
    reg64_t reg_kernel_base = rdx;
    reg64_t aux_reg_kernel = r9;
    reg64_t reg_output_base = rsi;
    reg64_t aux1_reg_input = reg_input_base;
    reg64_t aux1_reg_kernel = reg_output_base;

    reg64_t kj = r10;
    reg64_t oi_iter = r11;
    reg64_t reg_kh = abi_not_param1;
    reg64_t reg_overflow = reg_kh;
    reg64_t reg_oc_work = r14;
    reg64_t reg_table = r15;
    reg64_t reg_icb_iter = reg_oc_work;

    reg8_t reg_tmp_8 = r12b;
    reg16_t reg_tmp_16 = r12w;
    reg32_t reg_tmp_32 = r12d;
    reg64_t reg_tmp_64 = r12;

    reg64_t reg_d_weights = aux_reg_input;
    reg64_t reg_d_bias = aux_reg_kernel;
    reg64_t reg_oc_off = kj;
    reg64_t reg_tmp2_64 = reg_oc_off;
    reg32_t reg_tmp2_32 = reg_oc_off.cvt32();

    reg64_t reg_b_weights = aux_reg_input;
    reg64_t reg_b_mask = aux_reg_kernel;
    reg64_t reg_b_out_mask = reg_icb_iter;

    reg64_t reg_shift = aux_reg_input;

    Vmm vmm_scale = Vmm(isa == avx512_common ? 30 : 14);
    Vmm vmm_shift = Vmm(0);
    Vmm vmm_sum = Vmm(isa == avx512_common ? 26 : 10);
    Vmm vmm_lookup = Vmm(isa == avx512_common ? 28 : 12);
    Vmm vmm_mask = Vmm(isa == avx512_common ? 29 : 13);
    Vmm vmm_one_u8 = Vmm(isa == avx512_common ? 30 : 14);
    Vmm vmm_one_s16 = Vmm(isa == avx512_common ? 31 : 15);
    Ymm ymm_tmp = Ymm(isa == avx512_common ? 26 : 10);
    Vmm vmm_tmp = Vmm(isa == avx512_common ? 26 : 10);
    Vmm vmm_tmp1 = Vmm(isa == avx512_common ? 27 : 11);
    Vmm vmm_src = Vmm(0);
    Vmm vmm_tmp2 = Vmm(isa == avx512_common ? 25 : 9);
    Vmm vmm_thr = Vmm(isa == avx512_common ? 26 : 10);
    Vmm vmm_out_mask = Vmm(isa == avx512_common ? 30 : 14);

    const unsigned char _cmp_gt_os = 6;

    Xbyak::Opmask ktail_mask = Xbyak::Opmask(2);
    Xbyak::Opmask bin_mask0 = Xbyak::Opmask(5);
    Xbyak::Opmask bin_mask1 = Xbyak::Opmask(6);

    size_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Label l_table;

    nstl::vector<jit_uni_eltwise_injector_f32<isa>*> eltwise_injectors;
    nstl::vector<jit_uni_depthwise_injector_f32<isa>*> depthwise_injectors;

    inline void cvt2ps(data_type_t type_in, Vmm vmm_in, const Xbyak::Operand &op, bool scalar_load);
    inline void store_dst(const Xbyak::Address &op, Vmm vmm_dst, bool scalar_store);
    inline void apply_filter(int ur_w, int pad_l, int pad_r, int oc_blocks, int oc_step, int ic_blocks, bool last_icb, bool h_padded);
    inline void oh_step_unroll_kw(int ur_w, int pad_l, int pad_r, int oc_blocks, int oc_step, bool h_padded);
    inline void kh_loop(int ur_w, int pad_l, int pad_r, int oc_blocks, int oc_step);
    inline void width_blk_step(int ur_w, int pad_l, int pad_r, int oc_blocks, int oc_step);
    inline void solve_common(int oc_blocks, int oc_step);
    inline void prepare_table();

    void generate();
};

}
}
}

#endif
