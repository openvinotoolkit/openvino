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

#ifndef CPU_REF_DEFORMABLE_CONVOLUTION_HPP
#define CPU_REF_DEFORMABLE_CONVOLUTION_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_deformable_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct _ref_deformable_convolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public _cpu_deformable_convolution_fwd_pd_t {
        pd_t(engine_t *engine,
             const deformable_convolution_desc_t *adesc,
             const primitive_attr_t *attr,
             const typename pd_t::base_class *hint_fwd_pd)
                : _cpu_deformable_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
        {}

        DECLARE_COMMON_PD_T("ref:any", _ref_deformable_convolution_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace data_type;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                      && this->set_default_params() == status::success
                      && utils::one_of(this->desc()->prop_kind, forward_training,
                                       forward_inference)
                      && utils::one_of(this->desc()->alg_kind,
                                       alg_kind::deformable_convolution_direct)
                      && this->desc()->src_descs[0].data_type == f32
                      && this->desc()->src_descs[1].data_type == f32
                      && this->desc()->weights_desc.data_type == f32
                      && this->desc()->accum_data_type == f32
                      && this->desc()->dst_desc.data_type == f32
                      && IMPLICATION(this->with_bias(), this->desc()->bias_desc.data_type == f32)
                      && this->attr()->has_default_values();
            return ok ? status::success : status::unimplemented;
        }
    };

    _ref_deformable_convolution_fwd_t(const pd_t *apd, const input_vector &inputs,
                          const output_vector &outputs)
            : cpu_primitive_t(apd, inputs, outputs) {}

    typedef typename prec_traits<data_type::f32>::type src_data_t;
    typedef typename prec_traits<data_type::f32>::type wei_data_t;
    typedef typename prec_traits<data_type::f32>::type dst_data_t;
    typedef typename prec_traits<data_type::f32>::type acc_data_t;

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

using ref_deformable_convolution_fwd_t = _ref_deformable_convolution_fwd_t;

}
}
}

#endif
