/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#ifndef CPU_JIT_UNI_ROI_POOL_KERNEL_F32_HPP
#define CPU_JIT_UNI_ROI_POOL_KERNEL_F32_HPP

#include <cfloat>

#include "c_types_map.hpp"
#include "jit_generator.hpp"
#include "type_helpers.hpp"

#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

template <cpu_isa_t isa>
struct jit_uni_roi_pool_kernel_f32: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_roi_pool_kernel_f32)

    jit_uni_roi_pool_kernel_f32(jit_roi_pool_conf_t ajpp): jpp(ajpp)
    {
        this->generate();
        jit_ker = (decltype(jit_ker))this->getCode();
    }

    jit_roi_pool_conf_t jpp;

    void operator()(jit_roi_pool_call_s *arg) { jit_ker(arg); }
    static status_t init_conf(jit_roi_pool_conf_t &jbp,
            const roi_pooling_desc_t &pd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &dst_d);

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xmm, isa == avx2, Ymm, Zmm>::type;

    Vmm vmm_mask = Vmm(0);
    Vmm vmm_zero = Vmm(0);

    Xmm xmm_yf = Xmm(0);
    Vmm vmm_yf = Vmm(0);
    Xmm xmm_xf = Xmm(1);
    Vmm vmm_xf = Vmm(1);

    Vmm get_acc_reg(int idx) { return Vmm(2*idx + 1); }
    Vmm get_src_reg(int idx) { return Vmm(2*idx + 2); }

    Opmask k_store_mask = Opmask(7);

    const unsigned char _cmp_lt_os = 1;

    using reg64_t = const Xbyak::Reg64;
    reg64_t reg_input     = r8;
    reg64_t aux_reg_input = rax;
    reg64_t aux_reg_input1 = rdx;
    reg64_t reg_output    = r9;
    reg64_t reg_kh    = r10;
    reg64_t reg_kw    = r11;

    reg64_t h_iter = r14;
    reg64_t w_iter = r15;

    reg64_t reg_c_blocks = rbx;
    reg64_t reg_bin_area = rdx;

    reg64_t reg_yf = reg_kh;
    reg64_t reg_xf = reg_kw;

    reg64_t reg_yoff = h_iter;
    reg64_t reg_xoff = r12;

    void (*jit_ker)(jit_roi_pool_call_s *);

    void roi_pool_max(int c_blocks);
    void roi_pool_bilinear(int c_blocks);
    void empty_roi(int c_blocks);
    void loop_body(int c_blocks);

    void generate();
};

}
}
}

#endif
