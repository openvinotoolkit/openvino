/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#ifndef CPU_GEMM_INNER_PRODUCT_HPP
#define CPU_GEMM_INNER_PRODUCT_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_inner_product_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "gemm/gemm.hpp"
#include "gemm_inner_product_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct gemm_inner_product_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_inner_product_fwd_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : cpu_inner_product_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_inner_product_fwd_t);

        virtual status_t init() override {
            using namespace utils;
            assert(engine()->kind() == engine_kind::cpu);

            bool ok = true
                && this->set_default_params() == status::success
                && one_of(desc()->prop_kind, prop_kind::forward_training,
                        prop_kind::forward_inference)
                && !has_zero_dim_memory()
                && everyone_is(data_type, desc()->src_desc.data_type,
                        desc()->weights_desc.data_type,
                        desc()->dst_desc.data_type)
                && IMPLICATION(this->with_bias(),
                        data_type == desc()->bias_desc.data_type)
                && attr()->post_ops_.len_ <= 1
                && IMPLICATION(attr()->post_ops_.len_ == 1,
                        attr()->post_ops_.entry_[0].is_eltwise())
                && dense_gemm_consitency_check(src_pd(), weights_pd(),
                        dst_pd());
            return ok ? status::success : status::unimplemented;
        }
    };

    gemm_inner_product_fwd_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {
        bool has_bias = pd()->with_bias(),
             has_eltwise = pd()->attr()->post_ops_.len_ == 1,
             has_scale = !pd()->attr()->output_scales_.has_default_values();
        postops_in_ip_ = has_bias || has_eltwise || has_scale;
        pp_kernel_ = new inner_product_utils::pp_kernel_t<data_type, data_type>(
                apd);
    }
    ~gemm_inner_product_fwd_t() { delete pp_kernel_; }

    typedef typename prec_traits<data_type>::type data_t;

    virtual void execute(event_t *e) const {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    inner_product_utils::pp_kernel_t<data_type, data_type> *pp_kernel_;
    bool postops_in_ip_;
};

template <impl::data_type_t data_type>
struct gemm_inner_product_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_inner_product_bwd_data_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : cpu_inner_product_bwd_data_pd_t(engine, adesc, attr,
                    hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_inner_product_bwd_data_t);

        virtual status_t init() override {
            using namespace utils;
            assert(engine()->kind() == engine_kind::cpu);

            bool ok = true
                && this->set_default_params() == status::success
                && desc()->prop_kind == prop_kind::backward_data
                && !has_zero_dim_memory()
                && everyone_is(data_type, desc()->diff_src_desc.data_type,
                        desc()->weights_desc.data_type,
                        desc()->diff_dst_desc.data_type)
                && attr()->has_default_values()
                && dense_gemm_consitency_check(diff_src_pd(), weights_pd(),
                        diff_dst_pd());
            return ok ? status::success : status::unimplemented;
        }
    };

    gemm_inner_product_bwd_data_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual void execute(event_t *e) const {
        execute_backward_data();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_data() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

template <impl::data_type_t data_type>
struct gemm_inner_product_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public cpu_inner_product_bwd_weights_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : cpu_inner_product_bwd_weights_pd_t(engine, adesc, attr,
                    hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_inner_product_bwd_weights_t);

        virtual status_t init() override {
            using namespace utils;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && desc()->prop_kind == prop_kind::backward_weights
                && !has_zero_dim_memory()
                && everyone_is(data_type, desc()->src_desc.data_type,
                        desc()->diff_weights_desc.data_type,
                        desc()->diff_dst_desc.data_type)
                && attr()->has_default_values()
                && dense_gemm_consitency_check(src_pd(), diff_weights_pd(),
                        diff_dst_pd());

            return ok ? status::success : status::unimplemented;
        }
    };

    gemm_inner_product_bwd_weights_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual void execute(event_t *e) const {
        execute_backward_weights();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_weights() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

