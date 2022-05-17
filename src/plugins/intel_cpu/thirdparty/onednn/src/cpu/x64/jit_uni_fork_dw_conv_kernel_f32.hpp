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

#ifndef CPU_X64_JIT_UNI_FORK_DW_CONV_KERNEL_HPP
#define CPU_X64_JIT_UNI_FORK_DW_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_depthwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_quantization_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa>
struct jit_uni_fork_dw_conv_fwd_kernel_f32 : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_fork_dw_conv_fwd_kernel_f32)

    jit_uni_fork_dw_conv_fwd_kernel_f32(const jit_conv_conf_t &ajcp, const memory_desc_t &dst_md, const primitive_attr_t &attr)
            : jcp(ajcp), attr_(attr) {
    }

    ~jit_uni_fork_dw_conv_fwd_kernel_f32() {
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

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;

private:
    using Vmm = typename utils::conditional3<isa == sse41, Xbyak::Xmm,
        isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using mask_t = const Xbyak::Opmask;
    using reg64_t = const Xbyak::Reg64;
    const Xbyak::AddressFrame &vmmword = (isa == sse41)
        ? xword : (isa == avx2) ? yword : zword;
    const int vlen = cpu_isa_traits<isa>::vlen;

    // dw convolution
    reg64_t reg_input = r8;
    reg64_t aux_reg_input = r9;
    reg64_t aux1_reg_input = r10;
    reg64_t reg_kernel = r11;
    reg64_t aux_reg_kernel = r12;
    reg64_t reg_ch_blocks = r13;
    reg64_t reg_output = r14;
    reg64_t reg_bias = r15;
    reg64_t reg_tail = rax;
    reg64_t reg_kw = rbx;
    reg64_t iter_kh = rdx;
    reg64_t iter_kw = rsi;
    reg64_t reg_ur_w = rbp;
    reg64_t reg_kh = reg_tail;
    reg64_t aux1_reg_kernel = reg_ch_blocks;
    reg64_t imm_addr64 = aux1_reg_input;
    reg64_t aux_reg_ch_blocks = reg_ur_w;
    reg64_t aux_reg_blocks_offset = abi_not_param1;

    reg64_t reg_d_weights = imm_addr64;
    reg64_t reg_d_bias = iter_kh;
    int base_post_ops_data_offset = 0;
    constexpr static int reg64_size = 8;

    reg64_t reg_kd = aux_reg_blocks_offset;
    reg64_t aux_reg_inp_d = reg_input;
    reg64_t aux_reg_ker_d = reg_kernel;

    mask_t k_oc_tail_mask = Xbyak::Opmask(2);

    Vmm vmm_d_weights = Vmm(0);
    Vmm vmm_d_bias = Vmm(1);

    inline Vmm get_ker_reg(int idx) { return Vmm(idx + 0); }
    inline Vmm get_src_reg(int idx) { return Vmm(idx + 1); }
    inline Vmm get_acc_reg(int idx) { return Vmm(idx + 4); }

    inline bool is_src_layout_nxc() {
        return utils::one_of(jcp.src_tag, format_tag::ndhwc, format_tag::nhwc,
                             format_tag::nwc);
    }
    inline bool is_dst_layout_nxc() {
        return utils::one_of(jcp.dst_tag, format_tag::ndhwc, format_tag::nhwc,
                             format_tag::nwc);
    }

    inline void load_src(int ur_ch_blocks, int ur_w, bool is_ch_tail);
    inline void compute_loop(int ur_w, int ur_ch_blocks);
    inline void apply_filter(int ur_ch_blocks, int ur_w, bool is_ch_tail);
    inline void apply_filter_unrolled(int ur_ch_blocks, int ur_w, bool is_ch_tail);
    inline void apply_postprocess(int ur_ch_blocks, int ur_w);
    inline void store_dst(int ur_ch_blocks, int ur_w, bool is_ch_tail);
    inline void loop_body(int ur_ch_blocks);

    void load_tail(
            Vmm &vmm, const Xbyak::Reg64 &reg, int64_t offset, int load_size);
    void add_tail_from_mem(Vmm &vmm_acc, Vmm &vmm_tmp, const Xbyak::Reg64 &reg,
                           int64_t offset, int load_size);
    void store_tail(
            Vmm &vmm, const Xbyak::Reg64 &reg, int64_t offset, int store_size);

    void generate() override;

    nstl::vector<jit_uni_eltwise_injector_f32<isa>*> eltwise_injectors;
    nstl::vector<jit_uni_depthwise_injector_f32<isa>*> depthwise_injectors;
    nstl::vector<jit_uni_quantization_injector_f32<isa>*> quantization_injectors;
};

template <cpu_isa_t isa>
struct jit_uni_fork_dw_conv_bwd_data_kernel_f32: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_fork_dw_conv_bwd_data_kernel_f32)

    jit_uni_fork_dw_conv_bwd_data_kernel_f32(const jit_conv_conf_t &ajcp, const primitive_attr_t &attr)
            : jcp(ajcp), attr_(attr) {}

    ~jit_uni_fork_dw_conv_bwd_data_kernel_f32() {
        for (auto inj : depthwise_injectors)
            delete inj;
        depthwise_injectors.clear();
    }

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;

private:
    using Vmm = typename utils::conditional3<isa == sse41, Xbyak::Xmm,
        isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using reg64_t = const Xbyak::Reg64;

    inline Vmm get_ker_reg(int idx) { return Vmm(idx + 0); }
    inline Vmm get_src_reg(int idx) { return Vmm(idx + 1); }
    inline Vmm get_acc_reg(int idx) { return Vmm(idx + 4); }

    reg64_t reg_ddst       = rax;
    reg64_t aux_reg_ddst   = r8;
    reg64_t aux1_reg_ddst = abi_not_param1;
    reg64_t reg_kernel     = rdx;
    reg64_t aux_reg_kernel = r10;
    reg64_t aux1_reg_kernel = rbp;
    reg64_t reg_dsrc       = rsi;

    reg64_t reg_ur_str_w = r9;
    reg64_t reg_ch_blocks = rbx;

    reg64_t iter_kh = r11;
    reg64_t iter_kw = r12;
    reg64_t reg_kh  = r13;
    reg64_t reg_kw  = r14;

    reg64_t reg_d_weights = r15;
    reg64_t reg_d_bias = iter_kh;

    inline void loop_body(int ur_ch_blocks);
    inline void load_ddst(int ur_ch_blocks, int ur_str_w);
    inline void apply_filter(int ur_ch_blocks, int ur_str_w);
    inline void apply_postprocess(int ur_ch_blocks, int ur_str_w);
    inline void store_dsrc(int ur_ch_blocks, int ur_str_w);

    void generate() override;

    nstl::vector<jit_uni_depthwise_injector_f32<isa>*> depthwise_injectors;
};

}
}
}
}

#endif
