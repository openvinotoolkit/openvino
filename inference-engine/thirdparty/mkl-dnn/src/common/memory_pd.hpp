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

#ifndef MEMORY_PD_HPP
#define MEMORY_PD_HPP

#include <assert.h>
#include "mkldnn.h"

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "primitive_desc.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {

struct memory_pd_t: public primitive_desc_t {
    memory_pd_t(engine_t *engine)
        : primitive_desc_t(engine, primitive_kind::memory)
        , desc_(types::zero_md()) {}
    memory_pd_t(engine_t *engine, const memory_desc_t *adesc)
        : primitive_desc_t(engine, primitive_kind::memory)
        , desc_(*adesc)
    { assert(desc_.primitive_kind == kind()); }
    virtual ~memory_pd_t() {}

    inline const memory_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override
    { return reinterpret_cast<const op_desc_t *>(this->desc()); }

    virtual const memory_pd_t *input_pd(int index = 0) const override
    { return index == 0 ? this : nullptr; }
    virtual const memory_pd_t *output_pd(int index = 0) const override
    { return index == 0 ? this : nullptr; }
    virtual int n_inputs() const override { return 0; }
    virtual int n_outputs() const override { return 0; }

    virtual status_t query(query_t what, int idx, void *result) const override
    {
        switch (what) {
        case query::memory_d:
            *(const memory_desc_t**)result = desc();
            break;
        case query::memory_consumption_s64:
            *(ptrdiff_t*)result = get_size(); break;
        default:
            return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    inline bool is_zero() const
    { return memory_desc_wrapper(desc_).is_zero(); }

    virtual bool is_equal(const memory_pd_t *rhs) const {
        return engine_ == rhs->engine_
            && memory_desc_wrapper(desc_) == memory_desc_wrapper(*rhs->desc());
    }
    virtual size_t get_size() const
    { return memory_desc_wrapper(desc_).size(); }

    virtual status_t set_format(memory_format_t fmt) {
        memory_desc_t md = desc_;
        md.format = fmt;
        status_t status = memory_desc_wrapper::compute_blocking(md);
        if (status != status::success) return status;
        desc_ = md;
        return status;
    }

protected:
    memory_desc_t desc_;
};

struct view_pd_t: public primitive_desc_t {
    view_pd_t(engine_t *engine): primitive_desc_t(engine, primitive_kind::view)
    {}
    virtual ~view_pd_t() {}

    virtual const op_desc_t *op_desc() const override { return nullptr; }

    virtual const memory_pd_t *input_pd(int index = 0) const override
    { return index == 0 ? src_pd() : nullptr; }
    virtual const memory_pd_t *output_pd(int index = 0) const override
    { return index == 0 ? dst_pd() : nullptr; }
    virtual int n_inputs() const override { return 1; }
    virtual int n_outputs() const override { return 0; }
};

struct concat_pd_t: public primitive_desc_t {
    concat_pd_t(engine_t *engine, int n, int concat_dim,
            const primitive_attr_t *attr)
        : primitive_desc_t(engine, attr, primitive_kind::concat)
        , n_(n), concat_dim_(concat_dim) {}
    virtual ~concat_pd_t() {}

    virtual const op_desc_t *op_desc() const override { return nullptr; }
    virtual void init_info() override { init_info_mem(this, this->info_); }

    virtual const memory_pd_t *input_pd(int index = 0) const override
    { return index < n_inputs() ? src_pd(index) : nullptr; }
    virtual const memory_pd_t *output_pd(int index = 0) const override
    { return index == 0 ? dst_pd() : nullptr; }
    virtual int n_inputs() const override { return n_; }
    virtual int n_outputs() const override { return 1; }
    virtual int concat_dim() const { return concat_dim_; }
protected:
    int n_, concat_dim_;
};

struct sum_pd_t: public primitive_desc_t {
    sum_pd_t(engine_t *engine, int n, const primitive_attr_t *attr)
        : primitive_desc_t(engine, attr, primitive_kind::sum)
        , n_(n) {}
    virtual ~sum_pd_t() {}

    virtual const op_desc_t *op_desc() const override { return nullptr; }
    virtual void init_info() override { init_info_mem(this, this->info_); }

    virtual const memory_pd_t *input_pd(int index = 0) const override
    { return index < n_inputs() ? src_pd(index) : nullptr; }
    virtual const memory_pd_t *output_pd(int index = 0) const override
    { return index == 0 ? dst_pd() : nullptr; }
    virtual int n_inputs() const override { return n_; }
    virtual int n_outputs() const override { return 1; }
protected:
    int n_;
};

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
