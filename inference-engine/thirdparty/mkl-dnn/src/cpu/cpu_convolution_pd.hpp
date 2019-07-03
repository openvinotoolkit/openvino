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

#ifndef CPU_CONVOLUTION_FWD_PD_HPP
#define CPU_CONVOLUTION_FWD_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_memory.hpp"
#include "cpu_primitive.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct cpu_convolution_fwd_pd_t: public convolution_fwd_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_convolution_fwd_pd_t(engine_t *engine,
            const convolution_desc_t *adesc,
            const primitive_attr_t *attr,
            const typename cpu_convolution_fwd_pd_t::base_class *hint_fwd_pd)
        : convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
        , src_pd_(this->engine_, &this->desc()->src_desc)
        , dst_pd_(this->engine_, &this->desc()->dst_desc)
        , weights_pd_(this->engine_, &this->desc()->weights_desc)
        , bias_pd_(this->engine_, &this->desc()->bias_desc) {}
    virtual ~cpu_convolution_fwd_pd_t() {}

    virtual const cpu_memory_pd_t *src_pd(int index = 0) const override
    { return index == 0 ? &src_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *dst_pd(int index = 0) const override
    { return index == 0 ? &dst_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *weights_pd(int index = 0) const override {
        if (index == 0) return &weights_pd_;
        if (index == 1 && this->with_bias()) return &bias_pd_;
        return nullptr;
    }

    bool has_padded_dst() const {
        memory_desc_wrapper dst_d(&dst_pd_);
        if (!dst_d.is_blocking_desc()) return false;
        return this->OC() != dst_d.blocking_desc().padding_dims[1];
    }

    bool wants_padded_bias() const {
        if (!this->with_bias()) return false;
        return has_padded_dst();
    }

    bool wants_zero_pad_dst(bool jit_impl = true) const {
        if (!has_padded_dst()) return false;
        const auto &po = this->attr()->post_ops_;
        int idx;
        if ((idx = po.find(primitive_kind::eltwise)) == -1) return false;
        return !math::eltwise_fwd_preserves_zero(po.entry_[idx].eltwise.alg,
                jit_impl);
    }

protected:
    cpu_memory_pd_t src_pd_, dst_pd_;
    cpu_memory_pd_t weights_pd_, bias_pd_;

    inline memory_format_t src_format()
    {
        using namespace memory_format;
        return utils::pick(this->desc()->src_desc.ndims - 3, ncw, nchw, ncdhw);
    }
    inline memory_format_t wei_format()
    {
        using namespace memory_format;
        return this->with_groups()
            ? utils::pick(this->desc()->src_desc.ndims - 3, goiw, goihw, goidhw)
            : utils::pick(this->desc()->src_desc.ndims - 3, oiw, oihw, oidhw);
    }

    virtual status_t set_default_params() {
        using namespace memory_format;
        if (src_pd_.desc()->format == any)
            CHECK(src_pd_.set_format(src_format()));
        if (dst_pd_.desc()->format == any)
            CHECK(dst_pd_.set_format(src_pd_.desc()->format));
        if (weights_pd_.desc()->format == any)
            CHECK(weights_pd_.set_format(wei_format()));
        if (bias_pd_.desc()->format == any)
            CHECK(bias_pd_.set_format(x));
        if (this->desc()->alg_kind == alg_kind::convolution_auto)
            CHECK(this->set_alg_kind(alg_kind::convolution_direct));
        return status::success;
    }
};

struct cpu_convolution_bwd_data_pd_t: public convolution_bwd_data_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_convolution_bwd_data_pd_t(engine_t *engine,
            const convolution_desc_t *adesc,
            const primitive_attr_t *attr,
            const convolution_fwd_pd_t *hint_fwd_pd)
        : convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
        , diff_src_pd_(this->engine_, &this->desc_.diff_src_desc)
        , diff_dst_pd_(this->engine_, &this->desc_.diff_dst_desc)
        , weights_pd_(this->engine_, &this->desc_.weights_desc)
        , bias_pd_(this->engine_, &this->desc_.bias_desc) {}
    virtual ~cpu_convolution_bwd_data_pd_t() {}

    virtual const cpu_memory_pd_t *diff_src_pd(int index = 0) const override
    { return index == 0 ? &diff_src_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *diff_dst_pd(int index = 0) const override
    { return index == 0 ? &diff_dst_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *weights_pd(int index = 0) const override {
        if (index == 0) return &weights_pd_;
        if (index == 1 && this->with_bias()) return &bias_pd_;
        return nullptr;
    }

protected:
    cpu_memory_pd_t diff_src_pd_, diff_dst_pd_;
    cpu_memory_pd_t weights_pd_, bias_pd_;

    inline memory_format_t src_format()
    {
        using namespace memory_format;
        return utils::pick(this->desc_.diff_src_desc.ndims - 3, ncw, nchw, ncdhw);
    }
    inline memory_format_t wei_format()
    {
        using namespace memory_format;
        return this->with_groups()
            ? utils::pick(this->desc_.diff_src_desc.ndims - 3, goiw, goihw, goidhw)
            : utils::pick(this->desc_.diff_src_desc.ndims - 3, oiw, oihw, oidhw);
    }

    virtual status_t set_default_params() {
        using namespace memory_format;
        if (diff_src_pd_.desc()->format == any)
            CHECK(diff_src_pd_.set_format(src_format()));
        if (diff_dst_pd_.desc()->format == any)
            CHECK(diff_dst_pd_.set_format(diff_src_pd_.desc()->format));
        if (weights_pd_.desc()->format == any)
           CHECK(weights_pd_.set_format(wei_format()));
        if (bias_pd_.desc()->format == any)
            CHECK(bias_pd_.set_format(x));
        if (this->desc()->alg_kind == alg_kind::convolution_auto)
            CHECK(this->set_alg_kind(alg_kind::convolution_direct));
        return status::success;
    }
};

struct cpu_convolution_bwd_weights_pd_t: public convolution_bwd_weights_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_convolution_bwd_weights_pd_t(engine_t *engine,
            const convolution_desc_t *adesc,
            const primitive_attr_t *attr,
            const convolution_fwd_pd_t *hint_fwd_pd)
        : convolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
        , src_pd_(this->engine_, &this->desc_.src_desc)
        , diff_dst_pd_(this->engine_, &this->desc_.diff_dst_desc)
        , diff_weights_pd_(this->engine_, &this->desc_.diff_weights_desc)
        , diff_bias_pd_(this->engine_, &this->desc_.diff_bias_desc) {}
    virtual ~cpu_convolution_bwd_weights_pd_t() {}

    virtual const cpu_memory_pd_t *src_pd(int index = 0) const override
    { return index == 0 ? &src_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *diff_dst_pd(int index = 0) const override
    { return index == 0 ? &diff_dst_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *diff_weights_pd(int index = 0) const
        override {
            if (index == 0) return &diff_weights_pd_;
            if (index == 1 && this->with_bias()) return &diff_bias_pd_;
            return  nullptr;
        }

    bool wants_padded_bias() const {
        if (!this->with_bias()) return false;
        memory_desc_wrapper diff_dst_d(&diff_dst_pd_);
        if (!diff_dst_d.is_blocking_desc()) return false;
        return OC() != diff_dst_d.blocking_desc().padding_dims[1];
    }

protected:
    cpu_memory_pd_t src_pd_;
    cpu_memory_pd_t diff_dst_pd_;
    cpu_memory_pd_t diff_weights_pd_, diff_bias_pd_;

    inline memory_format_t src_format()
    {
        using namespace memory_format;
        return utils::pick(this->desc_.src_desc.ndims - 3, ncw, nchw, ncdhw);
    }
    inline memory_format_t wei_format()
    {
        using namespace memory_format;
        return this->with_groups()
            ? utils::pick(this->desc_.src_desc.ndims - 3, goiw, goihw, goidhw)
            : utils::pick(this->desc_.src_desc.ndims - 3, oiw, oihw, oidhw);
    }

    virtual status_t set_default_params() {
        using namespace memory_format;
        if (src_pd_.desc()->format == any)
            CHECK(src_pd_.set_format(src_format()));
        if (diff_dst_pd_.desc()->format == any)
            CHECK(diff_dst_pd_.set_format(src_format()));
        if (diff_weights_pd_.desc()->format == any)
            CHECK(diff_weights_pd_.set_format(wei_format()));
        if (diff_bias_pd_.desc()->format == any)
            CHECK(diff_bias_pd_.set_format(x));
        if (this->desc()->alg_kind == alg_kind::convolution_auto)
            CHECK(this->set_alg_kind(alg_kind::convolution_direct));
        return status::success;
    }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
