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

#ifndef JIT_UNI_DEF_CONV_KERNEL_F32_HPP
#define JIT_UNI_DEF_CONV_KERNEL_F32_HPP

#include "c_types_map.hpp"
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"
#include "cpu_memory.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
struct jit_uni_def_conv_fwd_kernel_f32: public jit_generator {
    jit_uni_def_conv_fwd_kernel_f32(jit_def_conv_conf_t ajcp,
            const primitive_attr_t &attr): jcp(ajcp), attr_(attr)
    {
        this->generate();
        jit_ker = (void (*)(jit_def_conv_call_s *))this->getCode();
    }

    ~jit_uni_def_conv_fwd_kernel_f32() {}

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_def_conv_fwd_kernel)

    static bool post_ops_ok(jit_def_conv_conf_t &jcp,
            const primitive_attr_t &attr);
    static status_t init_conf(jit_def_conv_conf_t &jcp,
            const deformable_convolution_desc_t &cd,
            cpu_memory_t::pd_t &src_pd,
            cpu_memory_t::pd_t &offsets_pd,
            cpu_memory_t::pd_t &weights_pd,
            cpu_memory_t::pd_t &dst_pd,
            cpu_memory_t::pd_t &bias_pd,
            const primitive_attr_t &attr);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_def_conv_conf_t &jcp, const primitive_attr_t &attr);

    jit_def_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_def_conv_call_s *);

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xbyak::Xmm,
            isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    const int vlen = cpu_isa_traits<isa>::vlen;
    using Ymm = const Xbyak::Ymm;
    using Xmm = const Xbyak::Xmm;
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using reg8_t = const Xbyak::Reg8;

    reg64_t reg_input = r8;
    reg64_t reg_def_off = r9;
    reg64_t reg_kernel = r10;
    reg64_t reg_bias = r11;
    reg64_t reg_output = r12;
    reg64_t reg_oh_pos = r13;
    reg64_t aux_reg_bias = rsi;
    reg64_t reg_ow_pos = rdx;
    reg64_t aux_reg_output = reg_ow_pos;
    reg64_t reg_dg_iter = reg_output;
    reg64_t aux_reg_input = rax;
    reg64_t aux2_reg_input = reg_kernel;
    reg64_t reg_ic_iter = rbx;
    reg64_t reg_oc_work = reg_ic_iter;
    reg64_t aux_reg_def_off = reg_bias;
    reg64_t reg_input_buffer = abi_not_param1;
    reg64_t aux_reg_input_buffer = r14;
    reg32_t reg_tmp_32 = r15d;
    reg64_t reg_tmp_64 = r15;
    reg64_t reg_table = rbp;
    reg64_t aux_reg_kernel = reg_table;
    reg64_t aux2_reg_kernel = r15;
    reg64_t aux2_reg_input_buffer = aux_reg_bias;
    reg64_t aux3_reg_input_buffer = reg_input;

    Xbyak::Opmask ktail_mask = Xbyak::Opmask(2);

    inline Xbyak::Address table_val(int index)
    { return ptr[reg_table + index * vlen]; }

    inline Vmm get_vmm_ker(int idx) { return Vmm(idx + 0); }
    inline Vmm get_vmm_src(int idx) { return Vmm(idx + 1); }
    inline Vmm get_vmm_acc(int idx) { return Vmm(idx + jcp.ur_w + 1); }
    inline Ymm get_ymm_acc(int idx) { return Ymm(idx + jcp.ur_w + 1); }
    inline Xmm get_xmm_acc(int idx) { return Xmm(idx + jcp.ur_w + 1); }

    inline void interpolate_input(int ow_step);
    inline void ic_loop(int ow_step, int oc_blocks_step, int oc_step);
    inline void store_output(int ow_step, int oc_blocks_step, int oc_step);
    inline void oc_loop(int ow_step);
    inline void ow_loop();
    inline void apply_filter(int ow_step, int oc_blocks_step, int oc_step, int ic_step);
    inline void init_accums(int ow_step, int oc_blocks_step, int oc_step);

    Xbyak::Label l_table;

    void generate();

    void prepare_table();
};

}
}
}

#endif
