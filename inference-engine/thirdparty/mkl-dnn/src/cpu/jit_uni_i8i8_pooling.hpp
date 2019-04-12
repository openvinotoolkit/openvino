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

#ifndef CPU_JIT_UNI_I8I8_POOLING_HPP
#define CPU_JIT_UNI_I8I8_POOLING_HPP

#include "c_types_map.hpp"
#include "cpu_isa_traits.hpp"
#include "cpu_pooling_pd.hpp"
#include "cpu_engine.hpp"

#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
struct jit_uni_i8i8_pooling_fwd_ker_t;

template <cpu_isa_t isa>
struct jit_uni_i8i8_pooling_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_pooling_fwd_pd_t {
        pd_t(engine_t *engine, const pooling_desc_t  *adesc,
                const primitive_attr_t *attr,
                const pooling_fwd_pd_t  *hint_fwd_pd)
        : cpu_pooling_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_uni_i8i8_pooling_fwd_t<isa>);

        virtual status_t init() override {
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && mayiuse(isa)
                && desc()->src_desc.ndims == 4
                && set_default_params() == status::success
                && desc()->prop_kind == prop_kind::forward_inference
                && utils::one_of(desc()->alg_kind, alg_kind::pooling_max,
                        alg_kind::pooling_avg_include_padding,
                        alg_kind::pooling_avg_exclude_padding)
                && utils::one_of(src_pd()->desc()->data_type, data_type::s32,
                        data_type::s8, data_type::u8)
                && src_pd()->desc()->data_type == dst_pd()->desc()->data_type
                && utils::everyone_is(memory_format::nhwc,
                        src_pd()->desc()->format, dst_pd()->desc()->format)
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            return jit_conf();
        }

        jit_pool_conf_t jpp_;

    protected:
        status_t jit_conf();

        virtual status_t set_default_params() override {
            using namespace memory_format;
            if (dst_pd_.desc()->format == any)
                CHECK(dst_pd_.set_format(nhwc));
            return status::success;
        }
    };

    jit_uni_i8i8_pooling_fwd_t(const pd_t *apd,
            const input_vector &inputs, const output_vector &outputs);
    ~jit_uni_i8i8_pooling_fwd_t();

    virtual void execute(event_t *e) const {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_uni_i8i8_pooling_fwd_ker_t<isa> *ker_;
};

}
}
}

#endif
