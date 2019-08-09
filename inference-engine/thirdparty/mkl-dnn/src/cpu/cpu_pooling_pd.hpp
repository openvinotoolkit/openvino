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

#ifndef CPU_POOLING_PD_HPP
#define CPU_POOLING_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "pooling_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_memory.hpp"
#include "cpu_primitive.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "nstl.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

inline data_type_t pooling_index_data_type(const pooling_desc_t *p) {
    using nstl::numeric_limits;
    /* the simplest way to express 256... */
    const int u8_max =
        numeric_limits<typename prec_traits<data_type::u8>::type>::max();
    /* value u8_max in the case of data_type::u8 is reserved for
       designation of invalid index when pooling window is fully placed
       outside of source domain */
    if( p->src_desc.ndims == 5 || p->diff_src_desc.ndims == 5 ) {
        return p->kernel[0] * p->kernel[1] * p->kernel[2] < u8_max
            ? data_type::u8 : data_type::s32;
    } else {
        return p->kernel[0] * p->kernel[1] < u8_max
            ? data_type::u8 : data_type::s32;
    }
}

struct cpu_pooling_fwd_pd_t: public pooling_fwd_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_pooling_fwd_pd_t(engine_t *engine, const pooling_desc_t *adesc,
            const primitive_attr_t *attr, const pooling_fwd_pd_t *hint_fwd_pd)
        : pooling_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
        , src_pd_(engine_, &desc_.src_desc), dst_pd_(engine_, &desc_.dst_desc)
        , ws_pd_(engine_) {}
    virtual ~cpu_pooling_fwd_pd_t() {}

    virtual const cpu_memory_pd_t *src_pd(int index = 0) const override
    { return index == 0 ? &src_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *dst_pd(int index = 0) const override
    { return index == 0 ? &dst_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *workspace_pd(int index = 0) const override
    { return (index == 0 && !ws_pd_.is_zero()) ? &ws_pd_ : nullptr; }

protected:
    cpu_memory_pd_t src_pd_;
    cpu_memory_pd_t dst_pd_;
    cpu_memory_pd_t ws_pd_;

    virtual status_t init() = 0;

    virtual status_t set_default_params() {
        using namespace memory_format;
        if (dst_pd_.desc()->format == any)
            CHECK(dst_pd_.set_format(src_pd_.desc()->format));
        return status::success;
    }
};

struct cpu_pooling_bwd_pd_t: public pooling_bwd_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_pooling_bwd_pd_t(engine_t *engine, const pooling_desc_t *adesc,
            const primitive_attr_t *attr, const pooling_fwd_pd_t *hint_fwd_pd)
        : pooling_bwd_pd_t(engine, adesc, attr, hint_fwd_pd)
        , diff_src_pd_(engine_, &desc_.diff_src_desc)
        , diff_dst_pd_(engine_, &desc_.diff_dst_desc)
        , ws_pd_(engine_) {}
    virtual ~cpu_pooling_bwd_pd_t() {}

    virtual const cpu_memory_pd_t *diff_src_pd(int index = 0) const override
    { return index == 0 ? &diff_src_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *diff_dst_pd(int index = 0) const override
    { return index == 0 ? &diff_dst_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *workspace_pd(int index = 0) const override
    { return (index == 0 && !ws_pd_.is_zero()) ? &ws_pd_ : nullptr; }

protected:
    cpu_memory_pd_t diff_src_pd_;
    cpu_memory_pd_t diff_dst_pd_;
    cpu_memory_pd_t ws_pd_;

    virtual status_t init() = 0;

    virtual status_t set_default_params() {
        using namespace memory_format;
        if (diff_src_pd_.desc()->format == any)
            CHECK(diff_src_pd_.set_format(diff_dst_pd_.desc()->format));
        return status::success;
    }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
