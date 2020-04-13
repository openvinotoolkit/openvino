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

#ifndef CPU_REF_LRN_FWD_HPP
#define CPU_REF_LRN_FWD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_lrn_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct ref_lrn_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_lrn_fwd_pd_t {
        pd_t(engine_t *engine, const lrn_desc_t *adesc,
                const primitive_attr_t *attr, const lrn_fwd_pd_t *hint_fwd_pd)
            : cpu_lrn_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ref:any", ref_lrn_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace alg_kind;
            using namespace data_type;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && utils::one_of(desc()->prop_kind, forward_training,
                        forward_inference)
                && utils::one_of(desc()->alg_kind, lrn_across_channels,
                        lrn_within_channel)
                && utils::everyone_is(data_type, desc()->data_desc.data_type)
                && IMPLICATION(data_type == bf16, mayiuse(avx512_core))
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            if (desc_.prop_kind == forward_training) { ws_pd_ = data_pd_; }

            return status::success;
        }
    };

    ref_lrn_fwd_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual void execute(event_t *e) const {
        using namespace memory_format;
        switch (pd()->src_pd()->desc()->format) {
        case nChw16c: execute_forward<nChw16c>(); break;
        case nChw8c: execute_forward<nChw8c>(); break;
        case nchw: execute_forward<nchw>(); break;
        case nhwc: execute_forward<nhwc>(); break;
        // XXX: fix compatibility with 0.14
        // mkldnn_any is used to call ref code for arbitrary format
        default: execute_forward<mkldnn_any>();
        }
        e->set_state(event_t::ready);
    }

private:
    template<memory_format_t fmt>void execute_forward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

template <impl::data_type_t data_type>
struct ref_lrn_bwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_lrn_bwd_pd_t {
        pd_t(engine_t *engine, const lrn_desc_t *adesc,
                const primitive_attr_t *attr, const lrn_fwd_pd_t *hint_fwd_pd)
            : cpu_lrn_bwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ref:any", ref_lrn_bwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace alg_kind;
            using namespace data_type;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && utils::one_of(desc()->prop_kind, backward_data)
                && utils::one_of(desc()->alg_kind, lrn_across_channels
                        /*, lrn_within_channel */) // not supported yet
                && utils::everyone_is(data_type, desc()->data_desc.data_type)
                && IMPLICATION(data_type == bf16,
                                    mayiuse(cpu_isa_t::avx512_core))
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            return status::success;
        }
    };

    ref_lrn_bwd_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual void execute(event_t *e) const {
        using namespace memory_format;
        switch (pd()->src_pd()->desc()->format) {
        case nChw16c: execute_backward<nChw16c>(); break;
        case nChw8c: execute_backward<nChw8c>(); break;
        case nchw: execute_backward<nchw>(); break;
        case nhwc: execute_backward<nhwc>(); break;
        // XXX: fix compatibility with 0.14
        // mkldnn_any is used to call ref code for arbitrary format
        default: execute_backward<mkldnn_any>();
        }
        e->set_state(event_t::ready);
    }

private:
    template<memory_format_t fmt>void execute_backward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
