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

#ifndef CPU_INNER_PRODUCT_FWD_PD_HPP
#define CPU_INNER_PRODUCT_FWD_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "inner_product_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_memory.hpp"
#include "cpu_primitive.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace {
inline bool dense_gemm_consitency_check(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &wei_d, const memory_desc_wrapper &dst_d) {
    using namespace memory_format;
    using namespace utils;
    return true
        && IMPLICATION(src_d.format() == nChw8c, wei_d.format() == oIhw8i)
        && IMPLICATION(src_d.format() == nChw16c, wei_d.format() == oIhw16i)
        && IMPLICATION(src_d.format() == nCdhw8c, wei_d.format() == oIdhw8i)
        && IMPLICATION(src_d.format() == nCdhw16c, wei_d.format() == oIdhw16i)
        && IMPLICATION(src_d.format() == nchw, wei_d.format() == oihw)
        && IMPLICATION(src_d.format() == ncdhw, wei_d.format() == oidhw)
        && IMPLICATION(src_d.format() == nhwc, wei_d.format() == hwio)
        && IMPLICATION(src_d.format() == ndhwc, wei_d.format() == dhwio)
        && IMPLICATION(src_d.format() == nc, one_of(wei_d.format(), oi, io))
        && dst_d.format() == nc
        && src_d.only_padded_dim(1)
        && wei_d.only_padded_dim(1)
        && src_d.blocking_desc().padding_dims[1]
            == wei_d.blocking_desc().padding_dims[1]
        && src_d.is_dense(true)
        && dst_d.is_dense()
        && wei_d.is_dense(true);
}
}

struct cpu_inner_product_fwd_pd_t: public inner_product_fwd_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_inner_product_fwd_pd_t(engine_t *engine,
            const inner_product_desc_t *adesc,
            const primitive_attr_t *attr,
            const inner_product_fwd_pd_t *hint_fwd_pd)
        : inner_product_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
        , src_pd_(engine_, &desc_.src_desc), dst_pd_(engine_, &desc_.dst_desc)
        , weights_pd_(engine_, &desc_.weights_desc)
        , bias_pd_(engine_, &desc_.bias_desc) {}
    virtual ~cpu_inner_product_fwd_pd_t() {}

    virtual const cpu_memory_pd_t *src_pd(int index = 0) const override
    { return index == 0 ? &src_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *dst_pd(int index = 0) const override
    { return index == 0 ? &dst_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *weights_pd(int index = 0) const override {
        if (index == 0) return &weights_pd_;
        if (index == 1 && with_bias()) return &bias_pd_;
        return nullptr;
    }

    int IC_total_padded() const {
        auto src_md = memory_desc_wrapper(src_pd());

        assert(src_md.is_blocking_desc());
        if (!src_md.is_blocking_desc()) return -1;

        return utils::array_product(src_md.blocking_desc().padding_dims + 1,
                ndims() - 1);
    }

protected:
    cpu_memory_pd_t src_pd_, dst_pd_;
    cpu_memory_pd_t weights_pd_, bias_pd_;

    virtual status_t set_default_params() {
        using namespace memory_format;
        if (src_pd_.desc()->format == any)
        {
            if (ndims() == 4) CHECK(src_pd_.set_format(nchw));
            else if (ndims() == 5) CHECK(src_pd_.set_format(ncdhw));
            else CHECK(src_pd_.set_format(nc));
        }
        if (dst_pd_.desc()->format == any)
            CHECK(dst_pd_.set_format(nc));
        if (weights_pd_.desc()->format == any)
        {
            if (ndims() == 4) CHECK(weights_pd_.set_format(oihw));
            else if (ndims() == 5) CHECK(weights_pd_.set_format(oidhw));
            else CHECK(weights_pd_.set_format(oi));
        }
        if (bias_pd_.desc()->format == any)
            CHECK(bias_pd_.set_format(x));
        return status::success;
    }
};

struct cpu_inner_product_bwd_data_pd_t: public inner_product_bwd_data_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_inner_product_bwd_data_pd_t(engine_t *engine,
            const inner_product_desc_t *adesc,
            const primitive_attr_t *attr,
            const inner_product_fwd_pd_t *hint_fwd_pd)
        : inner_product_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
        , diff_src_pd_(engine_, &desc_.diff_src_desc)
        , diff_dst_pd_(engine_, &desc_.diff_dst_desc)
        , weights_pd_(engine_, &desc_.weights_desc) {}
    virtual ~cpu_inner_product_bwd_data_pd_t() {}

    virtual const cpu_memory_pd_t *diff_src_pd(int index = 0) const override
    { return index == 0 ? &diff_src_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *diff_dst_pd(int index = 0) const override
    { return index == 0 ? &diff_dst_pd_ : nullptr; }
    virtual const cpu_memory_pd_t *weights_pd(int index = 0) const override
    { return index == 0 ? &weights_pd_ : nullptr; }

    int IC_total_padded() const {
        auto diff_src_md = memory_desc_wrapper(diff_src_pd());

        assert(diff_src_md.is_blocking_desc());
        if (!diff_src_md.is_blocking_desc()) return -1;

        return utils::array_product(
                diff_src_md.blocking_desc().padding_dims + 1, ndims() - 1);
    }

protected:
    cpu_memory_pd_t diff_src_pd_, diff_dst_pd_;
    cpu_memory_pd_t weights_pd_;

    virtual status_t set_default_params() {
        using namespace memory_format;
        if (diff_src_pd_.desc()->format == any)
        {
            if (ndims() == 4) CHECK(diff_src_pd_.set_format(nchw));
            else if (ndims() == 5) CHECK(diff_src_pd_.set_format(ncdhw));
            else CHECK(diff_src_pd_.set_format(nc));
        }
        if (diff_dst_pd_.desc()->format == any)
            CHECK(diff_dst_pd_.set_format(nc));
        if (weights_pd_.desc()->format == any)
        {
            if (ndims() == 4) CHECK(weights_pd_.set_format(oihw));
            else if (ndims() == 5) CHECK(weights_pd_.set_format(oidhw));
            else CHECK(weights_pd_.set_format(oi));
        }
        return status::success;
    }
};

struct cpu_inner_product_bwd_weights_pd_t: public inner_product_bwd_weights_pd_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    cpu_inner_product_bwd_weights_pd_t(engine_t *engine,
            const inner_product_desc_t *adesc,
            const primitive_attr_t *attr,
            const inner_product_fwd_pd_t *hint_fwd_pd)
        : inner_product_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
        , src_pd_(engine_, &desc_.src_desc)
        , diff_dst_pd_(engine_, &desc_.diff_dst_desc)
        , diff_weights_pd_(engine_, &desc_.diff_weights_desc)
        , diff_bias_pd_(engine_, &desc_.diff_bias_desc) {}
    virtual ~cpu_inner_product_bwd_weights_pd_t() {}

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

    int IC_total_padded() const {
        auto src_md = memory_desc_wrapper(src_pd());

        assert(src_md.is_blocking_desc());
        if (!src_md.is_blocking_desc()) return -1;

        return utils::array_product(src_md.blocking_desc().padding_dims + 1,
                ndims() - 1);
    }

protected:
    cpu_memory_pd_t src_pd_;
    cpu_memory_pd_t diff_dst_pd_;
    cpu_memory_pd_t diff_weights_pd_, diff_bias_pd_;

    virtual status_t set_default_params() {
        using namespace memory_format;
        if (src_pd_.desc()->format == any)
        {
            if (ndims() == 4) CHECK(src_pd_.set_format(nchw));
            else if (ndims() == 5) CHECK(src_pd_.set_format(ncdhw));
            else CHECK(src_pd_.set_format(nc));
        }
        if (diff_dst_pd_.desc()->format == any)
            CHECK(diff_dst_pd_.set_format(nc));
        if (diff_weights_pd_.desc()->format == any)
        {
            if (ndims() == 4) CHECK(diff_weights_pd_.set_format(oihw));
            else if (ndims() == 5) CHECK(diff_weights_pd_.set_format(oidhw));
            else CHECK(diff_weights_pd_.set_format(oi));
        }
        if (diff_bias_pd_.desc()->format == any)
            CHECK(diff_bias_pd_.set_format(x));
        return status::success;
    }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
