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

#ifndef SHUFFLE_PD_HPP
#define SHUFFLE_PD_HPP

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "memory_pd.hpp"

namespace mkldnn {
namespace impl {

struct shuffle_pd_t: public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::shuffle;

    typedef shuffle_pd_t base_class;
    typedef shuffle_pd_t hint_class;

    shuffle_pd_t(mkldnn::impl::engine_t *engine,
            const shuffle_desc_t *adesc,
            const primitive_attr_t *attr,
            const shuffle_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, attr, primitive_kind::shuffle)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd) {}
    virtual ~shuffle_pd_t() {}

    const shuffle_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override
    { return reinterpret_cast<const op_desc_t *>(this->desc()); }
    virtual void init_info() override { init_info_shuffle(this, this->info_); }

    virtual const memory_pd_t *input_pd(int index = 0) const override
    { return index == 0 ? (is_fwd() ? src_pd() : diff_dst_pd()) : nullptr; }
    virtual const memory_pd_t *output_pd(int index = 0) const override
    { return index == 0 ? (is_fwd() ? dst_pd() : diff_src_pd()) : nullptr; }

    virtual int n_inputs() const override { return 1; }
    virtual int n_outputs() const override { return 1; }

    virtual status_t query(query_t what, int idx, void *result) const override
    {
        switch (what) {
        case query::shuffle_d:
            *(const shuffle_desc_t**)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* shuffle aux functions */
    inline bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }
    inline int ndims() const { return desc_.data_desc.ndims; }
    inline int MB() const { return desc_.data_desc.dims[0]; }
    inline int C() const { return ndims() >= 2 ? desc_.data_desc.dims[1] : 1; }
    inline int D() const { return ndims() == 5 ? desc_.data_desc.dims[2] : 1; }
    inline int H() const { return ndims() >= 4 ?
                                  desc_.data_desc.dims[ndims() - 2] : 1; }
    inline int W() const { return ndims() >= 3 ?
                                  desc_.data_desc.dims[ndims() - 1] : 1; }
    inline int axis() const { return desc_.axis; }
    inline int axis_size() const { return desc_.data_desc.dims[axis()]; }
    inline int group_size() const { return desc_.group_size; }

protected:
    shuffle_desc_t desc_;
    const shuffle_pd_t *hint_fwd_pd_;
};

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
