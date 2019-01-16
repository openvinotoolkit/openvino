/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#ifndef CPU_JIT_UNI_ELTWISE_HPP
#define CPU_JIT_UNI_ELTWISE_HPP

#include <assert.h>
#include <mkldnn.hpp>

#include "c_types_map.hpp"
#include "cpu_eltwise_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
struct jit_uni_eltwise_injector_f32 {
    jit_uni_eltwise_injector_f32(jit_generator* host, alg_kind_t elt_alg_,
            float alpha_, float beta_, bool save_vecs_state_ = true,
            int table_reg_idx_ = 0, int opmask_idx_ = 1) {
        assert(utils::one_of(isa, sse42, avx2, avx512_common));
        assert(utils::one_of(elt_alg_, alg_kind::eltwise_relu,
                alg_kind::eltwise_tanh, alg_kind::eltwise_elu,
                alg_kind::eltwise_square, alg_kind::eltwise_abs,
                alg_kind::eltwise_sqrt, alg_kind::eltwise_linear,
                alg_kind::eltwise_bounded_relu, alg_kind::eltwise_soft_relu,
                alg_kind::eltwise_logistic, alg_kind::eltwise_clamp));

        h = host;
        elt_alg = elt_alg_;
        alpha = alpha_;
        beta = beta_;
        save_vecs_state = save_vecs_state_;
        table_reg_idx = table_reg_idx_;
        opmask_idx = opmask_idx_;
    }

    void compute_vector_range(size_t start_idx, size_t end_idx);
    void compute_vector(size_t idx);
    void prepare_table();

private:
    jit_generator* h;

    using Vmm = typename utils::conditional3<isa == sse42, Xbyak::Xmm,
            isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;

    size_t vlen = cpu_isa_traits<isa>::vlen;

    alg_kind_t elt_alg;
    float alpha;
    float beta;

    bool save_vecs_state;
    int table_reg_idx;
    int opmask_idx;

    const static size_t preserved_vecs_max = 5;

    size_t vecs_to_preserve = 0;
    size_t vecs_count = isa == avx512_common ? 32 : 16;
    size_t preserved_vecs_count = 0;
    size_t preserved_vec_idxs[preserved_vecs_max] = {0};
    size_t start_idx_tail = 0;

    Vmm vmm_mask, vmm_aux0, vmm_aux1, vmm_aux2, vmm_aux3;

    Xbyak::Reg64 p_table;
    Xbyak::Opmask k_mask;
    Xbyak::Label l_table;

    int aux_vecs_count(alg_kind_t elt_alg);

    void compute_body(size_t start_idx, size_t end_idx);
    void injector_preamble(size_t start_idx, size_t end_idx);
    void injector_preamble_tail(size_t start_idx);
    void injector_postamble();
    void assign_regs();
    bool is_free_vec(size_t idx);

    void exp_compute_vector(const Vmm &vmm_src);
    void relu_compute_vector(const Vmm &vmm_src);
    void relu_zero_ns_compute_vector(const Vmm &vmm_src);
    void elu_compute_vector(const Vmm &vmm_src);
    void tanh_compute_vector(const Vmm &vmm_src);
    void square_compute_vector(const Vmm &vmm_src);
    void abs_compute_vector(const Vmm &vmm_src);
    void sqrt_compute_vector(const Vmm &vmm_src);
    void linear_compute_vector(const Vmm &vmm_src);
    void bounded_relu_compute_vector(const Vmm &vmm_src);
    void soft_relu_compute_vector(const Vmm &vmm_src);
    void logistic_compute_vector(const Vmm &vmm_src);
    void clamp_compute_vector(const Vmm &vmm_src);

    void relu_prepare_table();
    void elu_prepare_table();
    void soft_relu_prepare_table();
    void abs_prepare_table();
    void sqrt_prepare_table();
    void linear_prepare_table();
    void bounded_relu_prepare_table();
    void clamp_prepare_table();
};

struct jit_uni_eltwise_kernel_f32;

template <cpu_isa_t isa>
struct jit_uni_eltwise_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_eltwise_fwd_pd_t {
        pd_t(engine_t *engine, const eltwise_desc_t *adesc,
                const primitive_attr_t *attr,
                const eltwise_fwd_pd_t *hint_fwd_pd)
            : cpu_eltwise_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_uni_eltwise_fwd_t<isa>);

        virtual status_t init() override;
    };

    jit_uni_eltwise_fwd_t(const pd_t *pd, const input_vector &inputs,
                       const output_vector &outputs);
    ~jit_uni_eltwise_fwd_t();

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e)
    {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    pd_t conf_;
    jit_uni_eltwise_kernel_f32 *kernel_;
};

template <cpu_isa_t isa>
struct jit_uni_eltwise_bwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_eltwise_bwd_pd_t {
        pd_t(engine_t *engine, const eltwise_desc_t *adesc,
                const primitive_attr_t *attr,
                const eltwise_fwd_pd_t *hint_fwd_pd)
            : cpu_eltwise_bwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_uni_eltwise_bwd_t<isa>);

        virtual status_t init() override;
    };

    jit_uni_eltwise_bwd_t(const pd_t *pd, const input_vector &inputs,
                       const output_vector &outputs);
    ~jit_uni_eltwise_bwd_t();

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e)
    {
        execute_backward();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward();
    pd_t conf_;
    jit_uni_eltwise_kernel_f32 *kernel_;
};

}
}
}

#endif
