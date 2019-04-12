/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#ifndef CPU_REF_ROI_POOLING_FWD_HPP
#define CPU_REF_ROI_POOLING_FWD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_roi_pooling_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct ref_roi_pooling_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_roi_pooling_fwd_pd_t {
        pd_t(engine_t *engine, const roi_pooling_desc_t *adesc,
                const primitive_attr_t *attr,
                const roi_pooling_fwd_pd_t *hint_fwd_pd)
        : cpu_roi_pooling_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ref:any", ref_roi_pooling_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
            && utils::one_of(desc()->prop_kind, forward_inference)
            && utils::everyone_is(data_type, src_pd()->desc()->data_type,
                                  dst_pd()->desc()->data_type);
            if (!ok) return status::unimplemented;

            return status::success;
        }
    };

    ref_roi_pooling_fwd_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) { }

    typedef typename prec_traits<data_type>::type data_t;

    ~ref_roi_pooling_fwd_t() { }

    virtual void execute(event_t *e) const {
        execute_forward_generic();
        e->set_state(event_t::ready);
    }
    
private:
    void execute_forward_generic() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
