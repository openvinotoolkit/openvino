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

#ifndef DECONVOLUTION_PD_HPP
#define DECONVOLUTION_PD_HPP

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "memory_pd.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"
namespace mkldnn {
namespace impl {

struct deconvolution_fwd_pd_t : public primitive_desc_t {
    typedef deconvolution_fwd_pd_t base_class;
    typedef deconvolution_fwd_pd_t hint_class;
    static constexpr auto base_pkind = primitive_kind::deconvolution;
    deconvolution_fwd_pd_t(mkldnn::impl::engine_t *engine,
            const deconvolution_desc_t *adesc, const primitive_attr_t *attr,
            const deconvolution_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, attr, base_pkind)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd) {}
    virtual ~deconvolution_fwd_pd_t() {}

    const deconvolution_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }
    virtual void init_info() override { init_info_conv(this, this->info_); }

    virtual const memory_pd_t *input_pd(int index = 0) const override {
        switch (index) {
        case 0: return src_pd();
        case 1:
        case 2: return weights_pd(index - 1);
        default: return nullptr;
        }
    }
    virtual const memory_pd_t *output_pd(int index = 0) const override {
        return index == 0 ? dst_pd() : nullptr;
    }

    virtual int n_inputs() const override { return 2 + with_bias(); }
    virtual int n_outputs() const override { return 1; }
    /* Memory format Query */
    virtual status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
        case pkind_traits<base_pkind>::query_d:
            *(const deconvolution_desc_t **)result = desc();
            break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common conv aux functions */
    inline int MB() const { return desc_.src_desc.dims[0]; }

    inline int IC() const { return desc_.src_desc.dims[1]; }
    inline int OC() const { return desc_.dst_desc.dims[1]; }
    inline int G() const
    { return with_groups() ? desc_.weights_desc.dims[0] : 1; }

    inline int ID() const { return (ndims() == 5)
        ? desc_.src_desc.dims[2] : 1; }
    inline int IH() const {
        return (ndims() == 3) ? 1 : desc_.src_desc.dims[ndims() - 2]; }
    inline int IW() const { return desc_.src_desc.dims[ndims()-1]; }
    inline int OD() const { return (ndims() == 5)
        ? desc_.dst_desc.dims[2] : 1; }
    inline int OH() const {
        return (ndims() == 3) ? 1 : desc_.dst_desc.dims[ndims() - 2]; }
    inline int OW() const { return desc_.dst_desc.dims[ndims()-1]; }
    inline int KD() const { return (ndims() == 5)
        ? desc_.weights_desc.dims[2 + with_groups()] : 1; }
    inline int KH() const { return (ndims() == 3)
            ? 1 : desc_.weights_desc.dims[ndims() - (2 - with_groups())]; }
    inline int KW() const
    { return desc_.weights_desc.dims[ndims() - (1 - with_groups())]; }
    inline int KSD() const { return (ndims() == 5) ? desc_.strides[0] : 1; }
    inline int KSH() const {
        return (ndims() == 3) ? 1 : desc_.strides[ndims() - 4]; }
    inline int KSW() const { return desc_.strides[ndims()-3]; }
    inline int KDD() const { return (ndims() == 5) ? desc_.dilates[0] : 0; }
    inline int KDH() const { return (ndims() == 3)
        ? 0 : desc_.dilates[ndims()-4]; }
    inline int KDW() const { return desc_.dilates[ndims()-3]; }
    inline int padFront() const
        { return (ndims() == 5) ? desc_.padding[0][0] : 0; }
    inline int padBack() const
        { return (ndims() == 5) ? desc_.padding[1][0] : 0; }
    inline int padT() const { return (ndims() == 3)
        ? 0 : desc_.padding[0][ndims()-4]; }
    inline int padB() const { return (ndims() == 3)
        ? 0 : desc_.padding[1][ndims()-4]; }
    inline int padL() const { return desc_.padding[0][ndims()-3]; }
    inline int padR() const { return desc_.padding[1][ndims()-3]; }

    inline bool with_bias() const {
        return !memory_desc_wrapper(desc_.bias_desc).is_zero();
    }
    inline bool with_groups() const {
        return desc_.weights_desc.ndims == desc_.src_desc.ndims + 1;
    }
    inline int ndims() const { return desc_.src_desc.ndims; }

    bool has_zero_dim_memory() const {
        return false
            || memory_desc_wrapper(desc_.src_desc).has_zero_dim()
            || memory_desc_wrapper(desc_.dst_desc).has_zero_dim();
    }

protected:
    deconvolution_desc_t desc_;
    const deconvolution_fwd_pd_t *hint_fwd_pd_;
    virtual status_t init() = 0;
};

struct deconvolution_bwd_data_pd_t : public primitive_desc_t {
    typedef deconvolution_bwd_data_pd_t base_class;
    typedef deconvolution_fwd_pd_t hint_class;
    static constexpr auto base_pkind = primitive_kind::deconvolution;

    deconvolution_bwd_data_pd_t(mkldnn::impl::engine_t *engine,
            const deconvolution_desc_t *adesc, const primitive_attr_t *attr,
            const deconvolution_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, attr, base_pkind)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd) {}
    virtual ~deconvolution_bwd_data_pd_t() {}

    const deconvolution_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }
    virtual void init_info() override { init_info_conv(this, this->info_); }

    virtual const memory_pd_t *input_pd(int index = 0) const override {
        switch (index) {
        case 0: return diff_dst_pd();
        case 1: return weights_pd(0);
        default: return nullptr;
        }
    }
    virtual const memory_pd_t *output_pd(int index = 0) const override {
        return index == 0 ? diff_src_pd() : nullptr;
    }

    virtual int n_inputs() const override { return 2; }
    virtual int n_outputs() const override { return 1; }

    virtual status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
        case query::deconvolution_d:
            *(const deconvolution_desc_t **)result = desc();
            break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common conv aux functions */
    inline int MB() const { return desc_.diff_src_desc.dims[0]; }

    inline int IC() const { return desc_.diff_src_desc.dims[1]; }
    inline int OC() const { return desc_.diff_dst_desc.dims[1]; }
    inline int G() const
    { return with_groups() ? desc_.weights_desc.dims[0] : 1; }

    inline int ID() const { return (ndims() == 5)
        ? desc_.diff_src_desc.dims[2] : 1; }
    inline int IH() const { return (ndims() == 3)
        ? 1 : desc_.diff_src_desc.dims[ndims()-2]; }
    inline int IW() const { return desc_.diff_src_desc.dims[ndims()-1]; }
    inline int OD() const { return (ndims() == 5)
        ? desc_.diff_dst_desc.dims[2] : 1; }
    inline int OH() const { return (ndims() == 3)
        ? 1 : desc_.diff_dst_desc.dims[ndims()-2]; }
    inline int OW() const { return desc_.diff_dst_desc.dims[ndims()-1]; }
    inline int KD() const { return (ndims() == 5)
        ? desc_.weights_desc.dims[2 + with_groups()] : 1; }
    inline int KH() const
    { return (ndims() == 3)
        ? 1 : desc_.weights_desc.dims[ndims() - (2 - with_groups())]; }
    inline int KW() const
    { return desc_.weights_desc.dims[ndims() - (1 - with_groups())]; }

    inline int KSD() const { return (ndims() == 5) ? desc_.strides[0] : 1; }
    inline int KSH() const { return (ndims() == 3)
        ? 1 : desc_.strides[ndims()-4]; }
    inline int KSW() const { return desc_.strides[ndims()-3]; }

    inline int KDD() const { return (ndims() == 5) ? desc_.dilates[0] : 0; }
    inline int KDH() const { return (ndims() == 3)
        ? 0 : desc_.dilates[ndims()-4]; }
    inline int KDW() const { return desc_.dilates[ndims()-3]; }

    inline int padFront() const
        { return (ndims() == 5) ? desc_.padding[0][0] : 0; }
    inline int padBack() const
        { return (ndims() == 5) ? desc_.padding[1][0] : 0; }
    inline int padT() const { return (ndims() == 3)
        ? 0 : desc_.padding[0][ndims()-4]; }
    inline int padB() const { return (ndims() == 3)
        ? 0 : desc_.padding[1][ndims()-4]; }
    inline int padL() const { return desc_.padding[0][ndims()-3]; }
    inline int padR() const { return desc_.padding[1][ndims()-3]; }

    inline bool with_bias() const {
        return !memory_desc_wrapper(desc_.bias_desc).is_zero();
    }
    inline bool with_groups() const {
        return desc_.weights_desc.ndims == desc_.diff_src_desc.ndims + 1;
    }
    inline int ndims() const { return desc_.diff_src_desc.ndims; }

protected:
    deconvolution_desc_t desc_;
    const deconvolution_fwd_pd_t *hint_fwd_pd_;
    virtual status_t init() = 0;
};

struct deconvolution_bwd_weights_pd_t : public primitive_desc_t {
    typedef deconvolution_bwd_weights_pd_t base_class;
    typedef deconvolution_fwd_pd_t hint_class;
    static constexpr auto base_pkind = primitive_kind::deconvolution;

    deconvolution_bwd_weights_pd_t(mkldnn::impl::engine_t *engine,
            const deconvolution_desc_t *adesc, const primitive_attr_t *attr,
            const deconvolution_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, attr, base_pkind)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd) {}
    virtual ~deconvolution_bwd_weights_pd_t() {}

    const deconvolution_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }
    virtual void init_info() override { init_info_conv(this, this->info_); }
    virtual const memory_pd_t *input_pd(int index = 0) const override {
        switch (index) {
        case 0: return src_pd();
        case 1: return diff_dst_pd();
        default: return nullptr;
        }
    }
    virtual const memory_pd_t *output_pd(int index = 0) const override {
        switch (index) {
        case 0: return diff_weights_pd(0);
        case 1: return with_bias() ? diff_weights_pd(1) : nullptr;
        default: return nullptr;
        }
    }

    virtual int n_inputs() const override { return 2; }
    virtual int n_outputs() const override { return 1 + with_bias(); }

    virtual status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
        case query::deconvolution_d:
            *(const deconvolution_desc_t **)result = desc();
            break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common conv aux functions */
    inline int MB() const { return desc_.src_desc.dims[0]; }

    inline int IC() const { return desc_.src_desc.dims[1]; }
    inline int OC() const { return desc_.diff_dst_desc.dims[1]; }
    inline int G() const
    { return with_groups() ? desc_.diff_weights_desc.dims[0] : 1; }

    inline int ID() const { return (ndims() == 5)
        ? desc_.src_desc.dims[2] : 1; }
    inline int IH() const { return (ndims() == 3)
        ? 1 : desc_.src_desc.dims[ndims()-2]; }
    inline int IW() const { return desc_.src_desc.dims[ndims()-1]; }
    inline int OD() const { return (ndims() == 5)
        ? desc_.diff_dst_desc.dims[2] : 1; }
    inline int OH() const { return (ndims() == 3)
        ? 1 : desc_.diff_dst_desc.dims[ndims()-2]; }
    inline int OW() const { return desc_.diff_dst_desc.dims[ndims()-1]; }
    inline int KD() const { return (ndims() == 5)
        ? desc_.diff_weights_desc.dims[2 + with_groups()] : 1; }
    inline int KH() const
    { return (ndims() == 3)
        ? 1 : desc_.diff_weights_desc.dims[ndims() - (2 - with_groups())]; }
    inline int KW() const
    { return desc_.diff_weights_desc.dims[ndims() - (1 - with_groups())]; }

    inline int KSD() const { return (ndims() == 5) ? desc_.strides[0] : 1; }
    inline int KSH() const { return (ndims() == 3)
        ? 1 : desc_.strides[ndims()-4]; }
    inline int KSW() const { return desc_.strides[ndims()-3]; }

    inline int KDD() const { return (ndims() == 5) ? desc_.dilates[0] : 0; }
    inline int KDH() const { return (ndims() == 3)
        ? 0 : desc_.dilates[ndims()-4]; }
    inline int KDW() const { return desc_.dilates[ndims()-3]; }

    inline int padFront() const
        { return (ndims() == 5) ? desc_.padding[0][0] : 0; }
    inline int padBack() const
        { return (ndims() == 5) ? desc_.padding[1][0] : 0; }
    inline int padT() const { return (ndims() == 3)
        ? 0 : desc_.padding[0][ndims()-4]; }
    inline int padB() const { return (ndims() == 3)
        ? 0 : desc_.padding[1][ndims()-4]; }
    inline int padL() const { return desc_.padding[0][ndims()-3]; }
    inline int padR() const { return desc_.padding[1][ndims()-3]; }

    inline bool with_bias() const {
        return !memory_desc_wrapper(desc_.diff_bias_desc).is_zero();
    }
    inline bool with_groups() const {
        return desc_.diff_weights_desc.ndims == desc_.diff_dst_desc.ndims + 1;
    }
    inline int ndims() const { return desc_.src_desc.ndims; }

protected:
    deconvolution_desc_t desc_;
    const deconvolution_fwd_pd_t *hint_fwd_pd_;
    virtual status_t init() = 0;
};
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
