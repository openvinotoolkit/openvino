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

#ifndef CPU_JIT_UNI_DEPTHWISE_HPP
#define CPU_JIT_UNI_DEPTHWISE_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_depthwise_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_generator.hpp"
#include "jit_uni_eltwise.hpp"
#include "jit_uni_quantization.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
struct jit_uni_depthwise_injector_f32 {
    jit_uni_depthwise_injector_f32(jit_generator* host, alg_kind_t depthwise_alg_, Xbyak::Opmask k_mask_ = Xbyak::Opmask(1))
        : h(host), depthwise_alg(depthwise_alg_), k_mask(k_mask_) {
        assert(utils::one_of(isa, sse42, avx2, avx512_common));
        assert(utils::one_of(depthwise_alg, alg_kind::depthwise_scale_shift, alg_kind::depthwise_prelu));
    }

    void compute_vector_range(int start_idx, int end_idx, const Xbyak::Reg64& p_weights, const Xbyak::Reg64& p_bias);

private:
    jit_generator* h;

    using Vmm = typename utils::conditional3<isa == sse42, Xbyak::Xmm,
            isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;

    size_t vlen = cpu_isa_traits<isa>::vlen;

    alg_kind_t depthwise_alg;

    Vmm vmm_mask;
    Vmm vmm_aux0;

    Xbyak::Opmask k_mask;

    const static size_t preserved_vecs_max = 5;
    size_t vecs_to_preserve = 0;
    size_t vecs_count = isa == avx512_common ? 32 : 16;
    size_t preserved_vecs_count = 0;
    size_t preserved_vec_idxs[preserved_vecs_max] = {0};
    size_t start_idx_tail = 0;

    int aux_vecs_count(alg_kind_t elt_alg);

    void compute_body(size_t start_idx, size_t end_idx, const Xbyak::Reg64& p_weights, const Xbyak::Reg64& p_bias);
    void injector_preamble(size_t start_idx, size_t end_idx);
    void injector_preamble_tail(size_t start_idx, size_t end_idx);
    void injector_postamble();
    void assign_regs();

    void scale_shift_compute_vector(const Vmm &vmm_src, const Xbyak::Reg64& p_weights, const Xbyak::Reg64& p_bias);
    void prelu_compute_vector(const Vmm &vmm_src, const Xbyak::Reg64& p_weights, const Xbyak::Reg64& p_bias);
};

struct jit_uni_depthwise_kernel_f32;

template <cpu_isa_t isa>
struct jit_uni_depthwise_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_depthwise_fwd_pd_t {
        pd_t(engine_t *engine, const depthwise_desc_t *adesc,
                const primitive_attr_t *attr,
                const depthwise_fwd_pd_t *hint_fwd_pd)
            : cpu_depthwise_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_uni_depthwise_fwd_t<isa>);

        virtual status_t init() override;
    };

    jit_uni_depthwise_fwd_t(const pd_t *apd, const input_vector &inputs,
                       const output_vector &outputs);
    ~jit_uni_depthwise_fwd_t();

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) const
    {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    jit_uni_depthwise_kernel_f32 *kernel_;
    data_t *padded_weights_;
    data_t *padded_bias_;
};


template <cpu_isa_t isa>
struct jit_uni_dw_conv_row_f32: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_ds_dw_conv_kernel_f32)

    jit_uni_dw_conv_row_f32(jit_conv_conf_t ajcp, const primitive_attr_t &attr, int ow_stride)
        : jcp(ajcp), attr_(attr), ow_stride_(ow_stride) {
        this->generate();
        jit_ker = (void (*)(jit_conv_call_s *))this->getCode();
    }

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
    static status_t init_conf(jit_conv_conf_t &jcp, jit_conv_conf_t &jcp_dw, const primitive_attr_t &attr);
    static status_t init_conf(jit_bin_conv_conf_t &jcp, jit_conv_conf_t &jcp_dw, const primitive_attr_t &attr);

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_conv_call_s *);
    int ow_stride_;

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xbyak::Xmm,
        isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using reg16_t = const Xbyak::Reg16;
    using reg8_t = const Xbyak::Reg8;
    const Xbyak::AddressFrame &vmmword = (isa == sse42)
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

    inline void load_src(int ur_w);
    inline void apply_filter(int ur_w, int kw_size);
    inline void cvt2ps(data_type_t type_in, Vmm vmm_in, const Xbyak::Operand &op, bool scalar_load);
    inline void apply_postprocessing(int ur_w, int oc_step);
    inline void store_dst_typed(const Xbyak::Address &op, Vmm vmm_dst, bool scalar_store);
    inline void store_dst(int ur_w, int oc_step);
    inline void loop_body(int oc_step);

    void generate();

    nstl::vector<jit_uni_eltwise_injector_f32<isa>*> eltwise_injectors;
    nstl::vector<jit_uni_depthwise_injector_f32<isa>*> depthwise_injectors;
    nstl::vector<jit_uni_quantization_injector_f32<isa>*> quantization_injectors;
};

}
}
}

#endif
