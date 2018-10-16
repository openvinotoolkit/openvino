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

#ifndef CPU_DEPTHWISE_PD_HPP
#define CPU_DEPTHWISE_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "depthwise_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_memory.hpp"
#include "cpu_primitive.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct cpu_depthwise_fwd_pd_t: public depthwise_fwd_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_depthwise_fwd_pd_t(engine_t *engine, const depthwise_desc_t *adesc,
            const primitive_attr_t *attr, const depthwise_fwd_pd_t *hint_fwd_pd)
        : depthwise_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
        , src_pd_(engine_, &desc_.src_desc)
        , dst_pd_(engine_, &desc_.dst_desc)
        , weights_pd_(engine_, &desc_.weights_desc)
        , bias_pd_(engine_, &desc_.bias_desc) {}
    virtual ~cpu_depthwise_fwd_pd_t() {}

    virtual const cpu_memory_pd_t *src_pd(int index = 0) const override
    { return index == 0 ? &src_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *dst_pd(int index = 0) const override
    { return index == 0 ? &dst_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *weights_pd(int index = 0) const override {
        if (index == 0) return &weights_pd_;
        if (index == 1 && this->with_bias()) return &bias_pd_;
        return nullptr;
    }

protected:
    cpu_memory_pd_t src_pd_, dst_pd_, weights_pd_, bias_pd_;

    virtual status_t set_default_params() {
        using namespace memory_format;
        if (src_pd_.desc()->format == any)
            CHECK(src_pd_.set_format(nchw));
        if (dst_pd_.desc()->format == any)
            CHECK(dst_pd_.set_format(nchw));
        if (weights_pd_.desc()->format == any)
            CHECK(weights_pd_.set_format(x));
        if (bias_pd_.desc()->format == any)
            CHECK(bias_pd_.set_format(x));
        return status::success;
    }

    virtual status_t init() = 0;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
