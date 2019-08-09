/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#ifndef CPU_JIT_UNI_SOFTMAX_HPP
#define CPU_JIT_UNI_SOFTMAX_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_softmax_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_uni_softmax_kernel_f32.hpp"
#include "mkldnn_types.h"


namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
struct jit_uni_softmax_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_softmax_fwd_pd_t {
        pd_t(engine_t *engine, const softmax_desc_t *adesc,
             const primitive_attr_t *attr,
             const softmax_fwd_pd_t *hint_fwd_pd)
            : cpu_softmax_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_uni_softmax_fwd_t<isa>);

        virtual status_t init() override {
            using namespace prop_kind;

            assert(engine()->kind() == engine_kind::cpu);

            auto ndims = desc_.data_desc.ndims;
            auto dims = desc_.data_desc.dims;
            auto axis = desc_.softmax_axis;

            size_t inner_size = utils::array_product(dims + axis + 1, ndims - axis - 1);

            memory_format_t desired_fmt;
            switch (ndims) {
                case 3: desired_fmt = memory_format::ncw; break;
                case 4: desired_fmt = memory_format::nchw; break;
                case 5: desired_fmt = memory_format::ncdhw; break;
                default: return status::unimplemented;
            }

            bool ok = mayiuse(isa)
                      && utils::one_of(desc()->prop_kind, forward_training,
                                       forward_inference)
                      && utils::everyone_is(data_type::f32, desc()->data_desc.data_type)
                      && memory_desc_wrapper(src_pd()).is_dense(true)
                      && utils::everyone_is(desired_fmt, src_pd()->desc()->format,
                                            dst_pd()->desc()->format)
                      && inner_size > 1;

            if (!ok) return status::unimplemented;


            return jit_uni_softmax_kernel_f32<isa>::init_conf(jpp_, desc_,
                                                              src_pd()->desc(), dst_pd()->desc());
        }
        jit_softmax_conf_t jpp_;
    };

    jit_uni_softmax_fwd_t(const pd_t *apd, const input_vector &inputs,
                       const output_vector &outputs);
    ~jit_uni_softmax_fwd_t();

    using data_t = prec_traits<data_type::f32>::type;

    virtual void execute(event_t *e) const override {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    jit_uni_softmax_kernel_f32<isa> *kernel_;
};

}
}
}

#endif
