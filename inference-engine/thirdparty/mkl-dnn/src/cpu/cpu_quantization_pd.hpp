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

#ifndef CPU_QUANTIZATION_PD_HPP
#define CPU_QUANTIZATION_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "quantization_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_memory.hpp"
#include "cpu_primitive.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct cpu_quantization_fwd_pd_t: public quantization_fwd_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_quantization_fwd_pd_t(engine_t *engine, const quantization_desc_t *adesc,
            const primitive_attr_t *attr, const quantization_fwd_pd_t *hint_fwd_pd)
        : quantization_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
        , src_pd_(engine_, &desc_.src_desc)
        , dst_pd_(engine_, &desc_.dst_desc)
        , thresholds_pd_(engine_, &desc_.thresholds_desc)
        , output_mask_pd_(engine_, &desc_.output_mask_desc)
        , crop_low_pd_(engine_, &desc_.crop_low_desc)
        , crop_high_pd_(engine_, &desc_.crop_high_desc)
        , input_scale_pd_(engine_, &desc_.input_scale_desc)
        , input_shift_pd_(engine_, &desc_.input_shift_desc)
        , output_scale_pd_(engine_, &desc_.output_scale_desc)
        , output_shift_pd_(engine_, &desc_.output_shift_desc) {}
    virtual ~cpu_quantization_fwd_pd_t() {}

    virtual const cpu_memory_pd_t *src_pd(int index = 0) const override
    { return index == 0 ? &src_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *dst_pd(int index = 0) const override
    { return index == 0 ? &dst_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *weights_pd(int index = 0) const override {
        if (is_binarization()) {
            if (index == 0) return &thresholds_pd_;
            if (index == 1) return &output_mask_pd_;
        } else {
            if (index == 0) return &crop_low_pd_;
            if (index == 1) return &crop_high_pd_;
            if (index == 2) return &input_scale_pd_;
            if (index == 3) return &input_shift_pd_;
            if (index == 4) return &output_scale_pd_;
            if (index == 5) return &output_shift_pd_;
        };
        return nullptr;
    }

protected:
    cpu_memory_pd_t src_pd_, dst_pd_, thresholds_pd_, output_mask_pd_, crop_low_pd_, crop_high_pd_,
                    input_scale_pd_, input_shift_pd_, output_scale_pd_, output_shift_pd_;

    inline memory_format_t src_format()
    {
        using namespace memory_format;
        return utils::pick(desc_.src_desc.ndims - 3, ncw, nchw, ncdhw);
    }
    inline memory_format_t wei_format()
    {
        using namespace memory_format;
        return x;
    }

    virtual status_t set_default_params() {
        using namespace memory_format;
        if (src_pd_.desc()->format == any)
            CHECK(src_pd_.set_format(src_format()));
        if (dst_pd_.desc()->format == any)
            CHECK(dst_pd_.set_format(src_pd_.desc()->format));
        if (thresholds_pd_.desc()->format == any)
            CHECK(thresholds_pd_.set_format(wei_format()));
        if (output_mask_pd_.desc()->format == any)
            CHECK(output_mask_pd_.set_format(wei_format()));
        if (crop_low_pd_.desc()->format == any)
            CHECK(crop_low_pd_.set_format(wei_format()));
        if (crop_high_pd_.desc()->format == any)
            CHECK(crop_high_pd_.set_format(wei_format()));
        if (input_scale_pd_.desc()->format == any)
            CHECK(input_scale_pd_.set_format(wei_format()));
        if (input_shift_pd_.desc()->format == any)
            CHECK(input_shift_pd_.set_format(wei_format()));
        if (output_scale_pd_.desc()->format == any)
            CHECK(output_scale_pd_.set_format(wei_format()));
        if (output_shift_pd_.desc()->format == any)
            CHECK(output_shift_pd_.set_format(wei_format()));
        return status::success;
    }

    virtual status_t init() = 0;
};

}
}
}

#endif
