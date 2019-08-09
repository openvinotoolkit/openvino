/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef CPU_DEFORMABLE_CONVOLUTION_FWD_PD_HPP
#define CPU_DEFORMABLE_CONVOLUTION_FWD_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "deformable_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_memory.hpp"
#include "cpu_primitive.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct _cpu_deformable_convolution_fwd_pd_t: public _deformable_convolution_fwd_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    _cpu_deformable_convolution_fwd_pd_t(engine_t *engine,
                             const deformable_convolution_desc_t *adesc,
                             const primitive_attr_t *attr,
                             const typename _cpu_deformable_convolution_fwd_pd_t::base_class *hint_fwd_pd)
            : _deformable_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , dst_pd_(this->engine_, &this->desc()->dst_desc)
            , weights_pd_(this->engine_, &this->desc()->weights_desc)
            , bias_pd_(this->engine_, &this->desc()->bias_desc) {
        for (int i = 0; i < 2; ++i) {
            src_pds_.push_back(cpu_memory_pd_t(engine_, &desc_.src_descs[i]));
        }
    }
    virtual ~_cpu_deformable_convolution_fwd_pd_t() {}

    virtual const cpu_memory_pd_t *src_pd(int index = 0) const override
    { return index < 2 ? &src_pds_[index] : nullptr; }
    virtual const cpu_memory_pd_t *dst_pd(int index = 0) const override
    { return index == 0 ? &dst_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *weights_pd(int index = 0) const override {
        if (index == 0) return &weights_pd_;
        if (index == 1 && this->with_bias()) return &bias_pd_;
        return nullptr;
    }

protected:
    nstl::vector<cpu_memory_pd_t> src_pds_;

    cpu_memory_pd_t dst_pd_;
    cpu_memory_pd_t weights_pd_, bias_pd_;

    inline memory_format_t src_format()
    {
        using namespace memory_format;
        return nchw;
    }
    inline memory_format_t wei_format()
    {
        using namespace memory_format;
        return this->with_groups() ? goihw : oihw;
    }

    virtual status_t set_default_params() {
        using namespace memory_format;
        if (src_pds_[0].desc()->format == any)
            CHECK(src_pds_[0].set_format(src_format()));
        if (src_pds_[1].desc()->format == any)
            CHECK(src_pds_[1].set_format(src_format()));
        if (dst_pd_.desc()->format == any)
            CHECK(dst_pd_.set_format(src_pds_[0].desc()->format));
        if (weights_pd_.desc()->format == any)
            CHECK(weights_pd_.set_format(wei_format()));
        if (bias_pd_.desc()->format == any)
            CHECK(bias_pd_.set_format(x));
        return status::success;
    }
};

using cpu_deformable_convolution_fwd_pd_t = _cpu_deformable_convolution_fwd_pd_t;

}
}
}

#endif
