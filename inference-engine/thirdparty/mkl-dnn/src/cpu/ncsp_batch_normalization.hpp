/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#ifndef CPU_NCSP_BATCH_NORMALIZATION_HPP
#define CPU_NCSP_BATCH_NORMALIZATION_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_batch_normalization_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct ncsp_batch_normalization_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_batch_normalization_fwd_pd_t {
        pd_t(engine_t *engine, const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : cpu_batch_normalization_fwd_pd_t(
                      engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ncsp_bnorm:any", ncsp_batch_normalization_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace data_type;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && is_fwd()
                && !has_zero_dim_memory()
                && desc()->data_desc.data_type == f32
                && IMPLICATION(use_scaleshift(),
                        desc()->data_scaleshift_desc.data_type == f32)
                && utils::one_of(data_pd_.desc()->format, memory_format::nchw,
                        memory_format::ncdhw, memory_format::nc)
                && (attr()->has_default_values() || this->with_relu_post_op());
            if (!ok)
                return status::unimplemented;

            if (is_training() && fuse_bn_relu()) {
                bn_init_default_ws(this, this->workspace_pd_, 8);
            }

            if (stats_is_src() || is_training()) {
                memory_desc_t stats_d;
                dims_t stats_dims = { C() };
                mkldnn_memory_desc_init(&stats_d, 1, stats_dims, data_type::f32,
                        memory_format::x);
                mean_pd_ = cpu_memory_t::pd_t(engine_, &stats_d);
                variance_pd_ = cpu_memory_t::pd_t(engine_, &stats_d);
            }

            return success;
        }
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    ncsp_batch_normalization_fwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs);
    ~ncsp_batch_normalization_fwd_t();

    virtual void execute(event_t *e) {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    data_t *stats_reduction_, *tmp_mean_, *tmp_variance_;
    void execute_forward();
    pd_t conf_;
};

struct ncsp_batch_normalization_bwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_batch_normalization_bwd_pd_t {
        pd_t(engine_t *engine, const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : cpu_batch_normalization_bwd_pd_t(
                      engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ncsp_bnorm:any", ncsp_batch_normalization_bwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace data_type;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && is_bwd()
                && !has_zero_dim_memory()
                && desc()->data_desc.data_type == f32
                && IMPLICATION(use_scaleshift(),
                        desc()->data_scaleshift_desc.data_type == f32)
                && utils::one_of(data_pd_.desc()->format, memory_format::nchw,
                        memory_format::ncdhw, memory_format::nc)
                && attr()->has_default_values();
            if (!ok)
                return status::unimplemented;

            if (fuse_bn_relu()) {
                bn_init_default_ws(this, this->workspace_pd_, 8);
                const size_t this_ws_sz
                        = memory_desc_wrapper(this->workspace_pd()).size();

                bool ws_ok = true && hint_fwd_pd_->workspace_pd()
                        && memory_desc_wrapper(hint_fwd_pd_->workspace_pd())
                                        .size()
                                == this_ws_sz;
                if (!ws_ok)
                    return status::unimplemented;
            }

            return success;
        }
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    ncsp_batch_normalization_bwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs);
    ~ncsp_batch_normalization_bwd_t();
    virtual void execute(event_t *e) {
        execute_backward();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward();
    pd_t conf_;

    data_t *stats_reduction_, *tmp_diff_scaleshift_;
};
}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
