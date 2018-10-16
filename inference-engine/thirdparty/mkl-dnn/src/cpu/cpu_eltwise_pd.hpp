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

#ifndef CPU_ELTWISE_PD_HPP
#define CPU_ELTWISE_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "eltwise_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_memory.hpp"
#include "cpu_primitive.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct cpu_eltwise_fwd_pd_t: public eltwise_fwd_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_eltwise_fwd_pd_t(engine_t *engine, const eltwise_desc_t *adesc,
            const primitive_attr_t *attr, const eltwise_fwd_pd_t *hint_fwd_pd)
        : eltwise_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
        , data_pd_(engine_, &desc_.data_desc) {}
    virtual ~cpu_eltwise_fwd_pd_t() {}

    virtual const cpu_memory_pd_t *src_pd(int index = 0) const override
    { return index == 0 ? &data_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *dst_pd(int index = 0) const override
    { return index == 0 ? &data_pd_ : nullptr; }

protected:
    cpu_memory_pd_t data_pd_;

    virtual status_t init() = 0;
};

struct cpu_eltwise_bwd_pd_t: public eltwise_bwd_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_eltwise_bwd_pd_t(engine_t *engine, const eltwise_desc_t *adesc,
            const primitive_attr_t *attr, const eltwise_fwd_pd_t *hint_fwd_pd)
        : eltwise_bwd_pd_t(engine, adesc, attr, hint_fwd_pd)
        , data_pd_(engine_, &desc_.data_desc)
        , diff_data_pd_(engine_, &desc_.diff_data_desc) {}
    virtual ~cpu_eltwise_bwd_pd_t() {}

    virtual const cpu_memory_pd_t *src_pd(int index = 0) const override
    { return index == 0 ? &data_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *diff_dst_pd(int index = 0) const override
    { return index == 0 ? &diff_data_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *diff_src_pd(int index = 0) const override
    { return index == 0 ? &diff_data_pd_ : nullptr; }

protected:
    cpu_memory_pd_t data_pd_;
    cpu_memory_pd_t diff_data_pd_;

    virtual status_t init() = 0;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
