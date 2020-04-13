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

#ifndef CPU_GEMM_INNER_PRODUCT_UTILS_HPP
#define CPU_GEMM_INNER_PRODUCT_UTILS_HPP

#include "c_types_map.hpp"
#include "cpu_inner_product_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "jit_generator.hpp"
#include "jit_uni_eltwise.hpp"
#include "jit_uni_depthwise.hpp"
#include "ref_eltwise.hpp"
#include "ref_depthwise.hpp"
#include "jit_avx512_core_bf16cvt.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace inner_product_utils {

template <impl::data_type_t acc_type, impl::data_type_t dst_type>
struct ker_args {
    typedef typename prec_traits<acc_type>::type acc_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    dst_data_t *dst;
    const acc_data_t *acc;
    const char *bias;
    const float *scales;
    size_t len;
    size_t oc_offset;
};

template <impl::data_type_t acc_type, impl::data_type_t dst_type>
class uni_pp_kernel_t {
public:
    virtual ~uni_pp_kernel_t() = default;

    void (*ker_)(const ker_args<acc_type, dst_type> *args);

    typedef typename prec_traits<acc_type>::type acc_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    virtual void operator()(dst_data_t *dst, const acc_data_t *acc, const char *bias,
                    const float *scales, size_t start, size_t end) = 0;
};

template <cpu_isa_t isa, impl::data_type_t acc_type, impl::data_type_t dst_type>
class jit_pp_kernel_t : public uni_pp_kernel_t<acc_type, dst_type>, jit_generator
{
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(gemm_x8s8s32x_inner_product_fwd_t::pp_kernel);
    jit_pp_kernel_t(const cpu_inner_product_fwd_pd_t *pd);
    ~jit_pp_kernel_t() {
        for (auto inj : eltwise_injectors_)
            delete inj;
        eltwise_injectors_.clear();
        for (auto inj : depthwise_injectors_)
            delete inj;
        depthwise_injectors_.clear();
    }

    typedef typename prec_traits<acc_type>::type acc_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    void operator()(dst_data_t *dst, const acc_data_t *acc, const char *bias,
            const float *scales, size_t start, size_t end) override;

private:
    void generate();

    enum {
        default_OC_loop_unroll_ = 4
    };

    void (*ker_)(const ker_args<acc_type, dst_type> *args);

    nstl::vector<jit_uni_eltwise_injector_f32<isa == avx512_core_bf16 ? avx512_common : isa> *> eltwise_injectors_;
    nstl::vector<jit_uni_depthwise_injector_f32<isa == avx512_core_bf16 ? avx512_common : isa> *> depthwise_injectors_;

    bf16_emulation_t *bf16_emu_;

    using Vmm = typename cpu_isa_traits<isa == avx512_core_bf16 ? avx512_common : isa>::Vmm;
    static const size_t vlen = cpu_isa_traits<isa == avx512_core_bf16 ? avx512_common : isa>::vlen / sizeof(float);

    Xbyak::Reg64 reg_param = abi_param1;
    Xbyak::Reg64 reg_dst = rdx;
    Xbyak::Reg64 reg_acc = rax;
    Xbyak::Reg64 reg_bias = rbx;
    Xbyak::Reg64 reg_scales = rsi;

    Xbyak::Reg64 reg_len = r8;
    Xbyak::Reg64 reg_tmp = rcx; // intentional for shifting purposes
    Xbyak::Reg64 reg_oc_offset = r9;
    Xbyak::Reg64 reg_rem_mask = r10;
    Xbyak::Opmask kreg_rem_mask = k1;

    Vmm vreg_zero, vreg_scale;

    //  dst_type == data_type::bf16 && isa != avx512_core_bf16
    Xbyak::Zmm bf16_emu_reserv_1 = Xbyak::Zmm(28);
    Xbyak::Zmm bf16_emu_reserv_2 = Xbyak::Zmm(29);
    Xbyak::Zmm bf16_emu_reserv_3 = Xbyak::Zmm(30);
    Xbyak::Reg64 bf16_emu_reserv_4 = r12;
    Xbyak::Zmm bf16_emu_reserv_5 = Xbyak::Zmm(31);

    //  sse42/avx2
    Xbyak::Reg64 reg_ptr_maskmovdqu_dst = rdi; // sse42: store destination - must be rdi
    Xbyak::Reg8 reg_tmp_8 = r11b;
    Xbyak::Reg32 reg_tmp_32 = r11d;
    Xbyak::Reg64 reg_tmp_64 = r11;
    Xbyak::Label l_table;
    Xbyak::Reg64 reg_table = r12;
    Xbyak::Reg64 reg_shift_table = r13;
    Vmm vreg_mask = Vmm(0); //  sse42: mask for blendvps must be in xmm0
    Vmm vreg_store_mask = Vmm(1);

    //  post_ops
    Xbyak::Opmask mask_post_op_reserved = k2;
    Xbyak::Reg64 eltwise_reserved = r11;
    Xbyak::Reg64 reg_d_weights = r14;
    Xbyak::Reg64 reg_d_bias = r15;
    Vmm vreg_d_weights, vreg_d_bias;
    post_ops_t post_ops_;

    size_t OC_;
    data_type_t bias_data_type_;
    size_t bias_data_type_size_;
    bool do_scale_;
    size_t scale_idx_mult_;
    round_mode_t rmode_;
    bool do_bias_;
    int max_OC_loop_unroll_;
    int idx_compute_vreg_start_;
    int idx_compute_vreg_max_;
    int compute_vregs_per_iter_;

    int idx_vreg_dst(int iter) {
        int idx = idx_compute_vreg_start_ + iter * compute_vregs_per_iter_ + 0;
        assert(idx <= idx_compute_vreg_max_);
        return idx;
    }
    int idx_vreg_bias(int iter) {
        int idx = idx_compute_vreg_start_ + iter * compute_vregs_per_iter_ + 1;
        assert(idx <= idx_compute_vreg_max_);
        return idx;
    }

    Vmm vreg_dst(int iter) { return Vmm(idx_vreg_dst(iter)); };
    Xbyak::Zmm zmm_dst(int iter) { return Xbyak::Zmm(idx_vreg_dst(iter)); };
    Xbyak::Ymm ymm_dst(int iter) { return Xbyak::Ymm(idx_vreg_dst(iter)); };
    Xbyak::Xmm xmm_dst(int iter) { return Xbyak::Xmm(idx_vreg_dst(iter)); };
    Vmm vreg_bias(int iter) { return Vmm(idx_vreg_bias(iter)); };
};

template <impl::data_type_t acc_type, impl::data_type_t dst_type>
class ref_pp_kernel_t : public uni_pp_kernel_t<acc_type, dst_type> {
public:
    ref_pp_kernel_t(const cpu_inner_product_fwd_pd_t *pd);
    ~ref_pp_kernel_t() {
        for (auto impl : ref_eltwise_impls_)
            delete impl;
        ref_eltwise_impls_.clear();
        for (auto impl : ref_depthwise_impls_)
            delete impl;
        ref_depthwise_impls_.clear();
    }

    typedef typename prec_traits<acc_type>::type acc_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    void operator()(dst_data_t *dst, const acc_data_t *acc, const char *bias,
                    const float *scales, size_t start, size_t end) override;

private:
    nstl::vector<ref_eltwise_scalar_fwd_t*> ref_eltwise_impls_;
    nstl::vector<ref_depthwise_scalar_fwd_t*> ref_depthwise_impls_;

    post_ops_t post_ops_;
    size_t OC_;
    data_type_t bias_data_type_;
    bool do_scale_;
    size_t scale_idx_mult_;
    round_mode_t rmode_;
    bool do_bias_;
};

}

}
}
}

#endif
