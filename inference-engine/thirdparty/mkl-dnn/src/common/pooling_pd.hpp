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

#ifndef POOLING_PD_HPP
#define POOLING_PD_HPP

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "memory_pd.hpp"

namespace mkldnn {
namespace impl {

struct pooling_fwd_pd_t: public primitive_desc_t {
    typedef pooling_fwd_pd_t base_class;
    typedef pooling_fwd_pd_t hint_class;
    static constexpr auto base_pkind = primitive_kind::pooling;

    pooling_fwd_pd_t(mkldnn::impl::engine_t *engine,
            const pooling_desc_t *adesc, const primitive_attr_t *attr,
            const pooling_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, attr, primitive_kind::pooling)
        , desc_(*adesc), hint_fwd_pd_(hint_fwd_pd) {}
    virtual ~pooling_fwd_pd_t() {}

    const pooling_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override
    { return reinterpret_cast<const op_desc_t *>(this->desc()); }
    virtual void init_info() override { init_info_pool(this, this->info_); }

    virtual const memory_pd_t *input_pd(int index = 0) const override
    { return index == 0 ? src_pd() : nullptr; }
    virtual const memory_pd_t *output_pd(int index = 0) const override {
        if (index == 0) return dst_pd();
        if (index == 1) return workspace_pd();
        return nullptr;
    }

    virtual int n_inputs() const override { return 1; }
    virtual int n_outputs() const override
    { return 1 + (workspace_pd() != nullptr); }

    virtual status_t query(query_t what, int idx, void *result) const override
    {
        switch (what) {
        case query::pooling_d:
            *(const pooling_desc_t**)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common pooling aux functions */
    inline bool is_3d() const { return input_pd()->desc()->ndims == 5; }

    inline int MB() const { return input_pd()->desc()->dims[0]; }
    inline int C() const { return input_pd()->desc()->dims[1]; }
    inline int ID() const { return is_3d() ? input_pd()->desc()->dims[2] : 1; }
    inline int IH() const { return is_3d()
        ? input_pd()->desc()->dims[3] : input_pd()->desc()->dims[2]; }
    inline int IW() const { return is_3d()
        ? input_pd()->desc()->dims[4] : input_pd()->desc()->dims[3]; }
    inline int OD() const { return is_3d()
        ? output_pd()->desc()->dims[2] : 1; }
    inline int OH() const { return is_3d()
        ? output_pd()->desc()->dims[3] : output_pd()->desc()->dims[2]; }
    inline int OW() const { return is_3d()
        ? output_pd()->desc()->dims[4] : output_pd()->desc()->dims[3]; }
    inline int KD() const { return is_3d() ? desc_.kernel[0] : 1; }
    inline int KH() const
    { return is_3d() ? desc_.kernel[1] : desc_.kernel[0]; }
    inline int KW() const
    { return is_3d() ? desc_.kernel[2] : desc_.kernel[1]; }

    inline int KSD() const { return is_3d() ? desc_.strides[0] : 1; }
    inline int KSH() const
    { return is_3d() ? desc_.strides[1] : desc_.strides[0]; }
    inline int KSW() const
    { return is_3d() ? desc_.strides[2] : desc_.strides[1]; }

    inline int padFront() const { return is_3d() ? desc_.padding[0][0] : 0; }
    inline int padBack() const { return is_3d() ? desc_.padding[1][0] : 0; }
    inline int padT() const { return is_3d()
        ? desc_.padding[0][1] : desc_.padding[0][0]; }
    inline int padB() const { return is_3d()
        ? desc_.padding[1][1] : desc_.padding[1][0]; }
    inline int padL() const { return is_3d()
        ? desc_.padding[0][2] : desc_.padding[0][1]; }
    inline int padR() const { return is_3d()
        ? desc_.padding[1][2] : desc_.padding[1][1]; }
protected:
    pooling_desc_t desc_;
    const pooling_fwd_pd_t *hint_fwd_pd_;
};

struct pooling_bwd_pd_t: public primitive_desc_t {
    typedef pooling_bwd_pd_t base_class;
    typedef pooling_fwd_pd_t hint_class;
    static constexpr auto base_pkind = primitive_kind::pooling;

    pooling_bwd_pd_t(mkldnn::impl::engine_t *engine,
            const pooling_desc_t *adesc, const primitive_attr_t *attr,
            const pooling_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, attr, primitive_kind::pooling)
        , desc_(*adesc), hint_fwd_pd_(hint_fwd_pd) {}
    virtual ~pooling_bwd_pd_t() {}

    const pooling_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override
    { return reinterpret_cast<const op_desc_t *>(this->desc()); }
    virtual void init_info() override { init_info_pool(this, this->info_); }

    virtual const memory_pd_t *input_pd(int index = 0) const override  {
        if (index == 0) return diff_dst_pd();
        if (index == 1) return workspace_pd();
        return nullptr;
    }
    virtual const memory_pd_t *output_pd(int index = 0) const override
    { return index == 0 ? diff_src_pd() : nullptr; }

    virtual int n_inputs() const override
    { return 1 + (workspace_pd() != nullptr); }
    virtual int n_outputs() const override { return 1; }

    virtual status_t query(query_t what, int idx, void *result) const override
    {
        switch (what) {
        case query::pooling_d:
            *(const pooling_desc_t**)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common pooling aux functions */

    inline bool is_3d() const { return desc_.diff_src_desc.ndims == 5; }

    inline int MB() const { return desc_.diff_src_desc.dims[0]; }
    inline int C() const { return desc_.diff_src_desc.dims[1]; }
    inline int ID() const { return is_3d() ? desc_.diff_src_desc.dims[2] : 1; }
    inline int IH() const { return is_3d()
        ? desc_.diff_src_desc.dims[3] : desc_.diff_src_desc.dims[2]; }
    inline int IW() const { return is_3d()
        ? desc_.diff_src_desc.dims[4] : desc_.diff_src_desc.dims[3]; }
    inline int OD() const { return is_3d()
        ? desc_.diff_dst_desc.dims[2] : 1; }
    inline int OH() const { return is_3d()
        ? desc_.diff_dst_desc.dims[3] : desc_.diff_dst_desc.dims[2]; }
    inline int OW() const { return is_3d()
        ? desc_.diff_dst_desc.dims[4] : desc_.diff_dst_desc.dims[3]; }
    inline int KD() const { return is_3d() ? desc_.kernel[0] : 1; }
    inline int KH() const
    { return is_3d() ? desc_.kernel[1] : desc_.kernel[0]; }
    inline int KW() const
    { return is_3d() ? desc_.kernel[2] : desc_.kernel[1]; }

    inline int KSD() const { return is_3d() ? desc_.strides[0] : 1; }
    inline int KSH() const
    { return is_3d() ? desc_.strides[1] : desc_.strides[0]; }
    inline int KSW() const
    { return is_3d() ? desc_.strides[2] : desc_.strides[1]; }

    inline int padFront() const { return is_3d() ? desc_.padding[0][0] : 0; }
    inline int padBack() const { return is_3d() ? desc_.padding[1][0] : 0; }
    inline int padT() const { return is_3d()
        ? desc_.padding[0][1] : desc_.padding[0][0]; }
    inline int padB() const { return is_3d()
        ? desc_.padding[1][1] : desc_.padding[1][0]; }
    inline int padL() const { return is_3d()
        ? desc_.padding[0][2] : desc_.padding[0][1]; }
    inline int padR() const { return is_3d()
        ? desc_.padding[1][2] : desc_.padding[1][1]; }

protected:
    pooling_desc_t desc_;
    const pooling_fwd_pd_t *hint_fwd_pd_;
};

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

