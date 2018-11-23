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

#ifndef CPU_BATCH_NORMALIZATION_FWD_PD_HPP
#define CPU_BATCH_NORMALIZATION_FWD_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "batch_normalization_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_memory.hpp"
#include "cpu_primitive.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace {
template <typename pd_t> inline void bn_init_default_ws(const pd_t *self,
        cpu_memory_t::pd_t &ws_pd, size_t bits_per_element) {
    memory_desc_t ws_d;
    const size_t src_nelems
        = memory_desc_wrapper(self->src_pd(0)).nelems(true);
    const size_t bits_per_byte = 8;
    const size_t ws_sz
        = utils::div_up(src_nelems * bits_per_element, bits_per_byte);
    dim_t mb = memory_desc_wrapper(self->src_pd(0)).dims()[0];
    dims_t ws_dims = { mb, (dim_t)( ws_sz / mb ) };
    mkldnn_memory_desc_init(&ws_d, 2, ws_dims, impl::data_type::u8,
            memory_format::nc);
    ws_pd = cpu_memory_t::pd_t(self->engine(), &ws_d);
}
}

struct cpu_batch_normalization_fwd_pd_t: public batch_normalization_fwd_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_batch_normalization_fwd_pd_t(engine_t *engine,
            const batch_normalization_desc_t *adesc,
            const primitive_attr_t *attr,
            const batch_normalization_fwd_pd_t *hint_fwd_pd)
        : batch_normalization_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
        , data_pd_(engine_, &desc_.data_desc)
        , mean_pd_(engine_)
        , variance_pd_(engine_)
        , scaleshift_pd_(engine_, &desc_.data_scaleshift_desc)
        , workspace_pd_(engine_) {}
    virtual ~cpu_batch_normalization_fwd_pd_t() {}

    virtual const cpu_memory_pd_t *src_pd(int index = 0) const override {
        if (index == 0) return &data_pd_;
        if (stats_is_src()) {
            if (index == 1) return &mean_pd_;
            if (index == 2) return &variance_pd_;
        }
        return nullptr;
    }

    virtual const cpu_memory_pd_t *dst_pd(int index = 0) const override {
        if (index == 0)  return &data_pd_;
        if (!stats_is_src() && is_training()) {
            if (index == 1) return &mean_pd_;
            if (index == 2) return &variance_pd_;
        }
        return nullptr;
    }

    virtual const cpu_memory_pd_t *weights_pd(int index = 0) const override
    { return index == 0 ? &scaleshift_pd_ : nullptr; }

    virtual const cpu_memory_pd_t *workspace_pd(int index = 0) const override {
        return index == 0 && is_training() && fuse_bn_relu()
            ? &workspace_pd_ : nullptr;
    }

protected:
    cpu_memory_pd_t data_pd_;
    cpu_memory_pd_t mean_pd_;
    cpu_memory_pd_t variance_pd_;
    cpu_memory_pd_t scaleshift_pd_;
    cpu_memory_pd_t workspace_pd_;

    virtual status_t init() = 0;
};

struct cpu_batch_normalization_bwd_pd_t: public batch_normalization_bwd_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_batch_normalization_bwd_pd_t(engine_t *engine,
            const batch_normalization_desc_t *adesc,
            const primitive_attr_t *attr,
            const batch_normalization_fwd_pd_t *hint_fwd_pd)
        : batch_normalization_bwd_pd_t(engine, adesc, attr, hint_fwd_pd)
        , data_pd_(engine_, &desc_.data_desc)
        , mean_pd_(engine_, &desc_.mean_desc)
        , variance_pd_(engine_, &desc_.variance_desc)
        , diff_data_pd_(engine_, &desc_.diff_data_desc)
        , scaleshift_pd_(engine_, &desc_.data_scaleshift_desc)
        , diff_scaleshift_pd_(engine_, &desc_.diff_data_scaleshift_desc)
        , workspace_pd_(engine_) {}
    virtual ~cpu_batch_normalization_bwd_pd_t() {}

    virtual const cpu_memory_pd_t *src_pd(int index = 0) const override {
        if (index == 0) return &data_pd_;
        if (index == 1) return &mean_pd_;
        if (index == 2) return &variance_pd_;

        return nullptr;
    }

    virtual const cpu_memory_pd_t *diff_dst_pd(int index = 0) const override
    { return index == 0 ? &diff_data_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *weights_pd(int index = 0) const override
    { return index == 0 ? &scaleshift_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *diff_weights_pd(int index = 0) const
        override { return index == 0 ? &diff_scaleshift_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *diff_src_pd(int index = 0) const override
    { return index == 0 ? &diff_data_pd_ : nullptr; }

    virtual const cpu_memory_pd_t *workspace_pd(int index = 0) const override
    { return index == 0 && fuse_bn_relu() ? &workspace_pd_ : nullptr; }

protected:
    cpu_memory_pd_t data_pd_;
    cpu_memory_pd_t mean_pd_;
    cpu_memory_pd_t variance_pd_;
    cpu_memory_pd_t diff_data_pd_;
    cpu_memory_pd_t scaleshift_pd_;
    cpu_memory_pd_t diff_scaleshift_pd_;
    cpu_memory_pd_t workspace_pd_;

    virtual status_t init() = 0;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
