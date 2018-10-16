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

#ifndef JIT_UNI_INNER_PRODUCT_HPP
#define JIT_UNI_INNER_PRODUCT_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_engine.hpp"
#include "cpu_inner_product_pd.hpp"
#include "gemm/jit_avx2_gemm_f32.hpp"
#include "gemm/jit_avx512_common_gemm_f32.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
struct jit_uni_inner_product_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_inner_product_fwd_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : cpu_inner_product_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
        {
        }

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("gemm:", isa, ""),
                jit_uni_inner_product_fwd_t<isa>);

        virtual status_t init() override
        {
            using namespace utils;

            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && mayiuse(isa)
                && this->set_default_params() == status::success
                && one_of(desc()->prop_kind, prop_kind::forward_training,
                        prop_kind::forward_inference)
                && everyone_is(data_type::f32, desc()->src_desc.data_type,
                        desc()->weights_desc.data_type,
                        desc()->dst_desc.data_type)
                && attr()->has_default_values()
                && implication(this->with_bias(),
                        data_type::f32 == desc()->bias_desc.data_type)
                && dense_gemm_consitency_check(src_pd(), weights_pd(),
                        dst_pd());
            return ok ? status::success : status::unimplemented;
        }
    };

    jit_uni_inner_product_fwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs);
    ~jit_uni_inner_product_fwd_t();

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e)
    {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    pd_t conf_;
    using jit_uni_gemm_f32 = typename utils::conditional
         <isa == avx2, jit_avx2_gemm_f32, jit_avx512_common_gemm_f32>::type;
    jit_uni_gemm_f32 *sgemm_;
};

template <cpu_isa_t isa>
struct jit_uni_inner_product_bwd_weights_t : public cpu_primitive_t {
    struct pd_t : public cpu_inner_product_bwd_weights_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : cpu_inner_product_bwd_weights_pd_t(engine, adesc, attr,
                    hint_fwd_pd)
        {
        }

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("gemm:", isa, ""),
                jit_uni_inner_product_bwd_weights_t<isa>);

        virtual status_t init() override
        {
            using namespace utils;
            assert(engine()->kind() == engine_kind::cpu);

            bool ok = true
                && mayiuse(isa)
                && this->set_default_params() == status::success
                && desc()->prop_kind == prop_kind::backward_weights
                && everyone_is(data_type::f32, desc()->src_desc.data_type,
                        desc()->diff_weights_desc.data_type,
                        desc()->diff_dst_desc.data_type)
                && attr()->has_default_values()
                && implication(this->with_bias(), data_type::f32
                        == desc()->diff_bias_desc.data_type)
                && dense_gemm_consitency_check(src_pd(), diff_weights_pd(),
                        diff_dst_pd());

            return ok ? status::success : status::unimplemented;
        }
    };

    jit_uni_inner_product_bwd_weights_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs);
    ~jit_uni_inner_product_bwd_weights_t();

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e)
    {
        execute_backward_weights();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_weights();
    pd_t conf_;
    using jit_uni_gemm_f32 = typename utils::conditional
        <isa == avx2, jit_avx2_gemm_f32, jit_avx512_common_gemm_f32>::type;
    jit_uni_gemm_f32 *sgemm_;
};

template <cpu_isa_t isa>
struct jit_uni_inner_product_bwd_data_t : public cpu_primitive_t {
    struct pd_t : public cpu_inner_product_bwd_data_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : cpu_inner_product_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
        {
        }

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("gemm:", isa, ""),
                jit_uni_inner_product_bwd_data_t<isa>);

        virtual status_t init() override
        {
            using namespace utils;
            assert(engine()->kind() == engine_kind::cpu);

            bool ok = true
                && mayiuse(isa)
                && this->set_default_params() == status::success
                && desc()->prop_kind == prop_kind::backward_data
                && everyone_is(data_type::f32,
                        desc()->diff_src_desc.data_type,
                        desc()->weights_desc.data_type,
                        desc()->diff_dst_desc.data_type)
                && attr()->has_default_values()
                && dense_gemm_consitency_check(diff_src_pd(), weights_pd(),
                        diff_dst_pd());
            return ok ? status::success : status::unimplemented;
        }
    };

    jit_uni_inner_product_bwd_data_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs);
    ~jit_uni_inner_product_bwd_data_t();

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e)
    {
        execute_backward_data();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_data();
    pd_t conf_;
    using jit_uni_gemm_f32 = typename utils::conditional
        <isa == avx2, jit_avx2_gemm_f32, jit_avx512_common_gemm_f32>::type;
    jit_uni_gemm_f32 *sgemm_;
};
}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
