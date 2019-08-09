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

#ifndef CPU_REF_POOLING_HPP
#define CPU_REF_POOLING_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_pooling_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type, impl::data_type_t acc_type = data_type>
struct ref_pooling_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_pooling_fwd_pd_t {
        pd_t(engine_t *engine, const pooling_desc_t *adesc,
                const primitive_attr_t *attr,
                const pooling_fwd_pd_t *hint_fwd_pd)
            : cpu_pooling_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ref:any", ref_pooling_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace alg_kind;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && set_default_params() == status::success
                && utils::one_of(desc()->prop_kind, forward_training,
                        forward_inference)
                && utils::one_of(desc()->alg_kind, pooling_max,
                        pooling_avg_include_padding,
                        pooling_avg_exclude_padding)
                && utils::everyone_is(data_type, src_pd()->desc()->data_type,
                        dst_pd()->desc()->data_type)
                && desc()->accum_data_type == acc_type
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            bool is_training = desc_.prop_kind == forward_training;
            if (desc()->alg_kind == pooling_max && is_training) {
                auto indices_desc = *dst_pd()->desc();
                indices_desc.data_type = pooling_index_data_type(desc());
                ws_pd_ = cpu_memory_t::pd_t(engine_, &indices_desc);
            }

            return status::success;
        }
    };

    ref_pooling_fwd_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {}

    typedef typename prec_traits<data_type>::type data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    virtual void execute(event_t *e) const {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

template <impl::data_type_t data_type>
struct ref_pooling_bwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_pooling_bwd_pd_t {
        pd_t(engine_t *engine, const pooling_desc_t *adesc,
                const primitive_attr_t *attr,
                const pooling_fwd_pd_t *hint_fwd_pd)
            : cpu_pooling_bwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ref:any", ref_pooling_bwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace alg_kind;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && set_default_params() == status::success
                && utils::one_of(desc()->prop_kind, backward_data)
                && utils::one_of(desc()->alg_kind, pooling_max,
                        pooling_avg_include_padding,
                        pooling_avg_exclude_padding)
                && utils::everyone_is(data_type, diff_dst_pd()->desc()->data_type,
                        diff_src_pd()->desc()->data_type)
                && IMPLICATION(desc()->alg_kind == pooling_max,
                        hint_fwd_pd_ && hint_fwd_pd_->workspace_pd()
                        && hint_fwd_pd_->workspace_pd()->engine()->kind()
                                == engine_kind::cpu)
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            if (desc()->alg_kind == pooling_max)
                ws_pd_ = *(cpu_memory_t::pd_t*)hint_fwd_pd_->workspace_pd();

            return status::success;
        }
    };

    ref_pooling_bwd_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual void execute(event_t *e) const {
        execute_backward();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
