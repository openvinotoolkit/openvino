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

#ifndef CPU_REF_INNER_PRODUCT_HPP
#define CPU_REF_INNER_PRODUCT_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_inner_product_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t src_type, impl::data_type_t wei_type = src_type,
         impl::data_type_t dst_type = src_type,
         impl::data_type_t acc_type = dst_type>
struct ref_inner_product_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_inner_product_fwd_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : cpu_inner_product_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ref:any", ref_inner_product_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace data_type;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(desc()->prop_kind, forward_training,
                        forward_inference)
                && desc()->src_desc.data_type == src_type
                && desc()->weights_desc.data_type == wei_type
                && desc()->accum_data_type == acc_type
                && desc()->dst_desc.data_type == dst_type
                && IMPLICATION(with_bias(),
                            utils::one_of(desc()->bias_desc.data_type,
                                f32, s32, s8, u8))
                && attr()->output_scales_.has_default_values()
                && attr()->post_ops_.len_ <= 1
                && IMPLICATION(attr()->post_ops_.len_ == 1,
                        attr()->post_ops_.entry_[0].is_relu(true, false));
            return ok ? status::success : status::unimplemented;
        }
    };

    ref_inner_product_fwd_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {}

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    virtual void execute(event_t *e) const {
        switch (pd()->desc()->prop_kind) {
        case prop_kind::forward_training:
        case prop_kind::forward_inference:
            execute_forward();
            break;
        default:
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    void execute_forward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

template <impl::data_type_t diff_src_type, impl::data_type_t wei_type,
         impl::data_type_t diff_dst_type,
         impl::data_type_t acc_type = diff_src_type>
struct ref_inner_product_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_inner_product_bwd_data_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : cpu_inner_product_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
        {}

        DECLARE_COMMON_PD_T("ref:any", ref_inner_product_bwd_data_t);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->desc()->prop_kind, backward,
                        backward_data)
                && desc()->diff_src_desc.data_type == diff_src_type
                && desc()->weights_desc.data_type == wei_type
                && desc()->accum_data_type == acc_type
                && desc()->diff_dst_desc.data_type == diff_dst_type
                && attr()->has_default_values();
            return ok ? status::success : status::unimplemented;
        }
    };

    ref_inner_product_bwd_data_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {}

    typedef typename prec_traits<diff_src_type>::type diff_src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<diff_dst_type>::type diff_dst_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    virtual void execute(event_t *e) const {
        switch (pd()->desc()->prop_kind) {
        case prop_kind::backward:
        case prop_kind::backward_data:
            execute_backward_data();
            break;
        default:
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_data() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

template <impl::data_type_t data_type>
struct ref_inner_product_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public cpu_inner_product_bwd_weights_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : cpu_inner_product_bwd_weights_pd_t(engine, adesc, attr,
                    hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ref:any", ref_inner_product_bwd_weights_t);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->desc()->prop_kind, backward,
                        backward_weights)
                && utils::everyone_is(data_type,
                        this->desc()->src_desc.data_type,
                        this->desc()->diff_dst_desc.data_type,
                        this->desc()->diff_weights_desc.data_type)
                && IMPLICATION(this->with_bias(),
                        data_type == this->desc()->diff_bias_desc.data_type)
                && attr()->has_default_values();
            return ok ? status::success : status::unimplemented;
        }
    };

    ref_inner_product_bwd_weights_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual void execute(event_t *e) const {
        switch (pd()->desc()->prop_kind) {
        case prop_kind::backward:
        case prop_kind::backward_weights:
            execute_backward_weights();
            break;
        default:
            assert(!"invalid prop_kind");
        }
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
