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

#ifndef CPU_JIT_UNI_BATCH_NORMALIZATION_HPP
#define CPU_JIT_UNI_BATCH_NORMALIZATION_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_batch_normalization_pd.hpp"
#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace { template <cpu_isa_t isa> struct uni_bnorm_driver_t; }

template <cpu_isa_t isa, data_type_t d_type>
struct jit_uni_batch_normalization_fwd_t : public cpu_primitive_t {
    struct pd_t: public cpu_batch_normalization_fwd_pd_t {
        pd_t(engine_t *engine, const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : cpu_batch_normalization_fwd_pd_t(engine, adesc, attr,
                    hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_uni_batch_normalization_fwd_t<isa, d_type>);

        virtual status_t init() override;
    };

    typedef typename prec_traits<d_type>::type data_t;

    jit_uni_batch_normalization_fwd_t(const pd_t *apd,
            const input_vector &inputs, const output_vector &outputs);
    ~jit_uni_batch_normalization_fwd_t();

    virtual void execute(event_t *e) const;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    uni_bnorm_driver_t<isa> *bnorm_driver_;
};

template <cpu_isa_t isa, data_type_t d_type>
struct jit_uni_batch_normalization_bwd_t : public cpu_primitive_t {
    struct pd_t: public cpu_batch_normalization_bwd_pd_t {
        pd_t(engine_t *engine, const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : cpu_batch_normalization_bwd_pd_t(engine, adesc, attr,
                    hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_uni_batch_normalization_bwd_t<isa, d_type>);

        virtual status_t init() override;
    };

    typedef typename prec_traits<d_type>::type data_t;

    jit_uni_batch_normalization_bwd_t(const pd_t *apd,
            const input_vector &inputs, const output_vector &outputs);
    ~jit_uni_batch_normalization_bwd_t();

    virtual void execute(event_t *e) const;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    uni_bnorm_driver_t<isa> *bnorm_driver_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
