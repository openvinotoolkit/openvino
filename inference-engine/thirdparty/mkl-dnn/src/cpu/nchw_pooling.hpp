/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

#ifndef CPU_NCHW_POOLING_HPP
#define CPU_NCHW_POOLING_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_pooling_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "bfloat16_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_format;

template <data_type_t d_type>
struct nchw_pooling_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_pooling_fwd_pd_t {
        pd_t(engine_t *engine, const pooling_desc_t *adesc,
                const primitive_attr_t *attr,
                const pooling_fwd_pd_t *hint_fwd_pd)
            : cpu_pooling_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("nchw_pooling:any", nchw_pooling_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace alg_kind;
            assert(engine()->kind() == engine_kind::cpu);
            auto src_format = src_pd()->desc()->format;
            bool ok = true
                && set_default_params() == status::success
                && utils::one_of(desc()->prop_kind, forward_training,
                        forward_inference)
                && utils::one_of(desc()->alg_kind, pooling_max,
                        pooling_avg_include_padding,
                        pooling_avg_exclude_padding)
                && !has_zero_dim_memory()
                && utils::everyone_is(d_type, src_pd()->desc()->data_type,
                        dst_pd()->desc()->data_type)
                && utils::one_of(src_format, nchw, ncdhw)
                && (src_format == dst_pd()->desc()->format)
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            bool is_training = desc_.prop_kind == forward_training;
            if (desc()->alg_kind == pooling_max && is_training) {
                auto indices_desc = *dst_pd()->desc();
                indices_desc.data_type = pooling_index_data_type(desc());
                ws_pd_ = cpu_memory_t::pd_t(engine_, &indices_desc);
            }

            init_scratchpad();

            return status::success;
        }

        private:
            void init_scratchpad() {
                using namespace memory_tracking::names;
                if (src_pd()->desc()->data_type == data_type::bf16) {
                    size_t src_sz_ = ID() * IH() * IW() * C() * MB();
                    auto scratchpad = scratchpad_registry().registrar();
                    scratchpad.book(key_pool_src_bf16cvt, sizeof(float) * src_sz_);
                }
            }
    };

    nchw_pooling_fwd_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {}

    ~nchw_pooling_fwd_t() {}

    typedef typename prec_traits<d_type>::type data_t;

    virtual void execute(event_t *e) const {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

template <data_type_t d_type>
struct nchw_pooling_bwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_pooling_bwd_pd_t {
        pd_t(engine_t *engine, const pooling_desc_t *adesc,
                const primitive_attr_t *attr,
                const pooling_fwd_pd_t *hint_fwd_pd)
            : cpu_pooling_bwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("nchw:any", nchw_pooling_bwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace alg_kind;
            assert(engine()->kind() == engine_kind::cpu);
            auto diff_dst_format = diff_dst_pd()->desc()->format;
            bool ok = true
                && set_default_params() == status::success
                && utils::one_of(desc()->prop_kind, backward_data)
                && utils::one_of(desc()->alg_kind, pooling_max,
                        pooling_avg_include_padding,
                        pooling_avg_exclude_padding)
                && !has_zero_dim_memory()
                && utils::everyone_is(d_type,
                        diff_dst_pd()->desc()->data_type,
                        diff_src_pd()->desc()->data_type)
                && utils::one_of(diff_dst_format, nchw, ncdhw)
                && (diff_dst_format == diff_src_pd()->desc()->format)
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            if (desc()->alg_kind == pooling_max) {
                bool ws_ok = true
                    && hint_fwd_pd_
                    && hint_fwd_pd_->workspace_pd()
                    && utils::one_of(
                            hint_fwd_pd_->workspace_pd()->desc()->format,
                            nchw, nChw8c, nChw16c, ncdhw, nCdhw8c, nCdhw16c);
                if (!ws_ok) return status::unimplemented;

                ws_pd_ = *(cpu_memory_t::pd_t*)hint_fwd_pd_->workspace_pd();
            }

            init_scratchpad();

            return status::success;
        }

        private:
            void init_scratchpad() {
                using namespace memory_tracking::names;
                if (diff_src_pd()->desc()->data_type == data_type::bf16) {
                    size_t dst_sz_ = OD() * OH() * OW();
                    size_t src_sz_ = ID() * IH() * IW();
                    size_t nthrs = mkldnn_get_max_threads();
                    auto scratchpad = scratchpad_registry().registrar();
                    scratchpad.book(key_pool_src_bf16cvt,
                            sizeof(float) * src_sz_ * nthrs);
                    scratchpad.book(key_pool_dst_bf16cvt,
                            sizeof(float) * dst_sz_ * nthrs);
                }
            }
    };

    nchw_pooling_bwd_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {}
    ~nchw_pooling_bwd_t() {}

    typedef typename prec_traits<d_type>::type data_t;

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
