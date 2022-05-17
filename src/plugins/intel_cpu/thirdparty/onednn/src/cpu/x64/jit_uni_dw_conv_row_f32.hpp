/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

/* [todo] antonvor:
 * This file contains the old plugin behavior in order to fix performance
 * problems after upgrading to OneDNN v1.6. This kernel is executed only on
 * machines with avx2 instruction set support and in the case of a fused
 * convolution. Remove after problems are fixed.
*/

#ifndef CPU_X64_JIT_UNI_DW_CONV_ROW_F32_HPP
#define CPU_X64_JIT_UNI_DW_CONV_ROW_F32_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "jit_primitive_conf.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_depthwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_quantization_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa>
struct jit_uni_dw_conv_row_f32: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_dw_conv_row_f32)

    jit_uni_dw_conv_row_f32(jit_conv_conf_t ajcp, const primitive_attr_t &attr, int ow_stride)
            : jcp(ajcp), attr_(attr), ow_stride_(ow_stride) {}

    ~jit_uni_dw_conv_row_f32() {
        for (auto inj : eltwise_injectors)
            delete inj;
        eltwise_injectors.clear();

        for (auto inj : depthwise_injectors)
            delete inj;
        depthwise_injectors.clear();

        for (auto inj : quantization_injectors)
            delete inj;
        quantization_injectors.clear();
    }

    static bool post_ops_ok(jit_conv_conf_t &jcp,
                            const primitive_attr_t &attr);
    static status_t init_conf(jit_1x1_conv_conf_t &jcp, jit_conv_conf_t &jcp_dw, const primitive_attr_t &attr);

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    int ow_stride_;

private:
    using Vmm = typename utils::conditional3<isa == sse41, Xbyak::Xmm,
            isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using reg16_t = const Xbyak::Reg16;
    using reg8_t = const Xbyak::Reg8;
    const Xbyak::AddressFrame &vmmword = (isa == sse41)
        ? xword : (isa == avx2) ? yword : zword;
    const int vlen = cpu_isa_traits<isa>::vlen;

    // dw convolution
    reg64_t reg_input0 = r8;
    reg64_t reg_input1 = r9;
    reg64_t reg_input2 = r10;
    reg64_t aux_reg_input0 = r11;
    reg64_t aux_reg_input1 = r12;
    reg64_t aux_reg_input2 = r13;

    reg64_t reg_kernel = r14;
    reg64_t aux_reg_kernel = r15;
    reg64_t reg_output = rdx;
    reg64_t reg_bias = rbx;
    reg64_t reg_kh = rax;
    reg64_t reg_ur_w = rbp;
    reg64_t reg_oc_work = abi_not_param1;

    reg64_t reg_oc_off = rsi;
    reg64_t reg_d_weights = aux_reg_input0;
    reg64_t reg_d_bias = aux_reg_input1;

    reg64_t reg_b_weights = r15;
    reg64_t reg_b_mask = reg_d_bias;
    reg64_t reg_b_out_mask = rbx;

    reg32_t reg_tmp_32 = r11d;
    reg64_t reg_tmp_64 = r11;
    reg8_t reg_tmp_8 = r11b;
    reg16_t reg_tmp_16 = r11w;

    reg32_t reg_tmp2_32 = r13d;
    reg64_t reg_tmp2_64 = r13;

    inline Vmm get_ker_reg(int idx) { return Vmm(idx + 0); }
    inline Vmm get_src_reg(int idx) { return Vmm(idx + 1); }
    inline Vmm get_acc_reg(int idx) { return Vmm(idx + 4); }

    Xbyak::Ymm ymm_tmp = Xbyak::Ymm(0);
    Vmm vmm_tmp = Vmm(0);
    Vmm vmm_sum = Vmm(0);
    Vmm vmm_bias = Vmm(0);
    Vmm vmm_thr = Vmm(0);
    Vmm vmm_out_mask = Vmm(1);

    Vmm vmm_d_weights = Vmm(0);
    Vmm vmm_d_bias = Vmm(1);

    const unsigned char _cmp_gt_os = 6;

    Xbyak::Opmask ktail_mask = Xbyak::Opmask(2);
    Xbyak::Opmask bin_mask0 = Xbyak::Opmask(5);
    Xbyak::Opmask bin_mask1 = Xbyak::Opmask(6);

    inline void clear_vmm_regs(int ur_w);
    inline void apply_filter(int ur_w, int kw_size);
    inline void cvt2ps(data_type_t type_in, Vmm vmm_in, const Xbyak::Operand &op, bool scalar_load);
    inline void apply_postprocessing(int ur_w, int oc_step);
    inline void store_dst_typed(const Xbyak::Address &op, Vmm vmm_dst, bool scalar_store);
    inline void store_dst(int ur_w, int oc_step);
    inline void loop_body(int oc_step);

    void generate() override;

    nstl::vector<jit_uni_eltwise_injector_f32<isa>*> eltwise_injectors;
    nstl::vector<jit_uni_depthwise_injector_f32<isa>*> depthwise_injectors;
    nstl::vector<jit_uni_quantization_injector_f32<isa>*> quantization_injectors;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
