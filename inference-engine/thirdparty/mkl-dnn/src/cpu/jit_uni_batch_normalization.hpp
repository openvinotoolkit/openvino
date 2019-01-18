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
#include "cpu_batch_normalization_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace { template <cpu_isa_t isa> struct uni_bnorm_driver_t; }

template <cpu_isa_t isa>
struct jit_uni_batch_normalization_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_batch_normalization_fwd_pd_t {
        pd_t(engine_t *engine, const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : cpu_batch_normalization_fwd_pd_t(engine, adesc, attr,
                    hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_uni_batch_normalization_fwd_t<isa>);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace data_type;
            using namespace memory_format;
            assert(engine()->kind() == engine_kind::cpu);
            auto desired_fmt = (ndims() == 4)
                ? isa == avx512_common ? nChw16c : nChw8c
                : isa == avx512_common ? nCdhw16c : nCdhw8c;
            bool ok = true
                && mayiuse(isa)
                && is_fwd()
                && !has_zero_dim_memory()
                && utils::one_of(ndims(), 4, 5)
                && desc()->data_desc.data_type == f32
                && IMPLICATION(use_scaleshift(),
                        desc()->data_scaleshift_desc.data_type == f32)
                && desc()->data_desc.format == desired_fmt
                && (attr()->has_default_values() || this->with_relu_post_op());
            if (!ok) return status::unimplemented;

            if (is_training() && fuse_bn_relu()) {
                if (isa < avx2) return status::unimplemented;
                bn_init_default_ws(this, this->workspace_pd_, 1);
            }
            if (memory_desc_wrapper(&data_pd_).blocking_desc()
                .padding_dims[1] != this->C() && isa < avx2)
                return status::unimplemented;

            if (stats_is_src() || is_training()) {
                memory_desc_t stats_d;
                dims_t stats_dims = { C() };
                mkldnn_memory_desc_init(&stats_d, 1, stats_dims,
                        data_type::f32, x);
                mean_pd_ = cpu_memory_t::pd_t(engine_, &stats_d);
                variance_pd_ = cpu_memory_t::pd_t(engine_, &stats_d);
            }

            return success;
        }
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    jit_uni_batch_normalization_fwd_t(const pd_t *pd,
            const input_vector &inputs, const output_vector &outputs);
    ~jit_uni_batch_normalization_fwd_t();
    virtual void execute(event_t *e);

private:
    uni_bnorm_driver_t<isa> *bnorm_driver_;

    pd_t conf_;
};

template <cpu_isa_t isa>
struct jit_uni_batch_normalization_bwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_batch_normalization_bwd_pd_t {
        pd_t(engine_t *engine, const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : cpu_batch_normalization_bwd_pd_t(engine, adesc, attr,
                    hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_uni_batch_normalization_bwd_t<isa>);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace data_type;
            using namespace utils;
            using namespace memory_format;
            assert(engine()->kind() == engine_kind::cpu);
            auto desired_fmt = (ndims() == 4)
                ? utils::one_of(isa, sse42, avx2) ? nChw8c : nChw16c
                : utils::one_of(isa, sse42, avx2) ? nCdhw8c : nCdhw16c;
            bool ok = true
                && mayiuse(isa)
                && is_bwd()
                && !has_zero_dim_memory()
                && utils::one_of(ndims(), 4, 5)
                && everyone_is(f32, desc()->data_desc.data_type,
                        desc()->diff_data_desc.data_type)
                && IMPLICATION(use_scaleshift(),
                        desc()->data_scaleshift_desc.data_type == f32)
                && everyone_is(desired_fmt, desc()->diff_data_desc.format,
                        desc()->data_desc.format)
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;
            if (memory_desc_wrapper(&data_pd_).blocking_desc()
                .padding_dims[1] != this->C() && isa < avx2)
                return status::unimplemented;

            if (fuse_bn_relu()) {
                if (isa < avx2) return status::unimplemented;
                bn_init_default_ws(this, this->workspace_pd_, 1);
                const size_t this_ws_sz
                    = memory_desc_wrapper(this->workspace_pd()).size();

                bool ws_ok = true
                    && hint_fwd_pd_->workspace_pd()
                    && memory_desc_wrapper(hint_fwd_pd_->workspace_pd()).size()
                            == this_ws_sz;
                if (!ws_ok)
                    return status::unimplemented;
            }

            /* TODO: extra checks required */

            return success;
        }
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    jit_uni_batch_normalization_bwd_t(const pd_t *pd,
            const input_vector &inputs, const output_vector &outputs);
    ~jit_uni_batch_normalization_bwd_t();
    virtual void execute(event_t *e);

private:
    uni_bnorm_driver_t<isa> *bnorm_driver_;

    pd_t conf_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
