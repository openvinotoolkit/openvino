/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#ifndef CPU_ROI_POOLING_FWD_PD_HPP
#define CPU_ROI_POOLING_FWD_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "roi_pooling_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_memory.hpp"
#include "cpu_primitive.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct cpu_roi_pooling_fwd_pd_t: public roi_pooling_fwd_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_roi_pooling_fwd_pd_t(engine_t *engine, const roi_pooling_desc_t *adesc,
            const primitive_attr_t *attr,
            const roi_pooling_fwd_pd_t *hint_fwd_pd)
        : roi_pooling_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
        , num_src_(desc_.num_src)
        , dst_pd_(engine_, &desc_.dst_desc)
    {
        for (int i = 0; i < num_src_; ++i) {
            src_pds_.push_back(cpu_memory_pd_t(engine_, &desc_.src_desc[i]));
        }
    }

    virtual ~cpu_roi_pooling_fwd_pd_t() {}

    virtual const cpu_memory_pd_t *src_pd(int index = 0) const override {
        return &src_pds_[index];
    }

    virtual const cpu_memory_pd_t *dst_pd(int index = 0) const override
    { return index == 0 ? &dst_pd_ : nullptr; }

    int num_src() {
        return num_src_;
    }

protected:
    nstl::vector<cpu_memory_pd_t> src_pds_;
    int num_src_;

    cpu_memory_pd_t dst_pd_;

    virtual status_t init() = 0;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
