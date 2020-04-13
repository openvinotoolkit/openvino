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

#ifndef CPU_REF_QUANTIZATION_HPP
#define CPU_REF_QUANTIZATION_HPP

#include <assert.h>

#include "cpu_quantization_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "c_types_map.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t src_type, impl::data_type_t dst_type>
struct ref_quantization_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_quantization_fwd_pd_t {
        pd_t(engine_t *engine, const quantization_desc_t *adesc,
                const primitive_attr_t *attr,
                const quantization_fwd_pd_t *hint_fwd_pd)
            : cpu_quantization_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ref:any", ref_quantization_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(engine()->kind() == engine_kind::cpu);

            bool ok = true
                && utils::one_of(desc()->prop_kind, forward_training, forward_inference)
                && utils::one_of(desc()->alg_kind, mkldnn_quantization_quantize_dequantize, mkldnn_quantization_quantize, mkldnn_binarization_depthwise)
                && utils::everyone_is(src_type, desc()->src_desc.data_type)
                && IMPLICATION(utils::one_of(desc()->alg_kind, mkldnn_quantization_quantize_dequantize, mkldnn_quantization_quantize),
                        utils::everyone_is(mkldnn_f32, desc()->crop_low_desc.data_type, desc()->crop_high_desc.data_type,
                                                       desc()->input_scale_desc.data_type, desc()->input_shift_desc.data_type,
                                                       desc()->output_scale_desc.data_type, desc()->output_shift_desc.data_type)
                        && desc()->src_desc.ndims > 0 && desc()->src_desc.ndims <= 5)
                && IMPLICATION(desc()->alg_kind == mkldnn_binarization_depthwise,
                               utils::everyone_is(mkldnn_f32, desc()->thresholds_desc.data_type, desc()->output_mask_desc.data_type)
                               && desc()->src_desc.ndims > 1 && desc()->src_desc.ndims <= 5)
                && desc()->src_desc.ndims == desc()->dst_desc.ndims
                && utils::everyone_is(dst_type, desc()->dst_desc.data_type)
                && utils::one_of(axis(), 0, 1)
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            return status::success;
        }
    };

    ref_quantization_fwd_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {}

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    virtual void execute(event_t *e) const {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

}
}
}

#endif
