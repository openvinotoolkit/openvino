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

#ifndef BINARY_CONVOLUTION_PD_HPP
#define BINARY_CONVOLUTION_PD_HPP

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "memory_pd.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {

status_t bin_conv_desc_init(binary_convolution_desc_t *bin_conv_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *dst_desc,
        const dims_t strides, const dims_t dilates,
        const dims_t padding_l, const dims_t padding_r,
        padding_kind_t padding_kind);

struct _binary_convolution_fwd_pd_t: public primitive_desc_t {
    typedef _binary_convolution_fwd_pd_t base_class;
    typedef _binary_convolution_fwd_pd_t hint_class;
    typedef binary_convolution_desc_t base_desc_t;
    static constexpr auto base_pkind = primitive_kind::binary_convolution;

    _binary_convolution_fwd_pd_t(mkldnn::impl::engine_t *engine,
            const base_desc_t *adesc, const primitive_attr_t *attr,
            const _binary_convolution_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, attr, base_pkind), desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd) {}
    virtual ~_binary_convolution_fwd_pd_t() {}

    const base_desc_t *desc() const { return &desc_; }
    inline const binary_convolution_desc_t *cdesc() const { return &cdesc_(); }
    virtual const op_desc_t *op_desc() const override
    { return reinterpret_cast<const op_desc_t *>(this->desc()); }
    virtual void init_info() override { init_info_bin_conv(this, this->info_); }

    virtual const memory_pd_t *input_pd(int index = 0) const override {
        switch (index) {
        case 0: return src_pd();
        case 1: return weights_pd(index - 1);
        default: return nullptr;
        }
    }
    virtual const memory_pd_t *output_pd(int index = 0) const override
    { return index == 0 ? dst_pd() : nullptr; }

    virtual int n_inputs() const override { return 2; }
    virtual int n_outputs() const override { return 1; }

    virtual status_t query(query_t what, int idx, void *result) const override
    {
        switch (what) {
        case pkind_traits<base_pkind>::query_d:
            *(const base_desc_t**)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common conv aux functions */

    inline int MB() const { return input_pd()->desc()->dims[0]; }

    inline int IC() const { return input_pd()->desc()->dims[1]; }
    inline int OC() const { return output_pd()->desc()->dims[1]; }
    inline int G() const
    { return with_groups() ? cdesc_().weights_desc.dims[0] : 1; }

    inline int ID() const { return (ndims() == 5) ? input_pd()->desc()->dims[2] : 1; }
    inline int IH() const { return (ndims() == 3) ? 1 : input_pd()->desc()->dims[ndims()-2]; }
    inline int IW() const { return input_pd()->desc()->dims[ndims()-1]; }
    inline int OD() const { return (ndims() == 5) ? output_pd()->desc()->dims[2] : 1; }
    inline int OH() const { return (ndims() == 3) ? 1 : output_pd()->desc()->dims[ndims()-2]; }
    inline int OW() const { return output_pd()->desc()->dims[ndims()-1]; }
    inline int KD() const { return (ndims() == 5)
        ? cdesc_().weights_desc.dims[2 + with_groups()] : 1; }
    inline int KH() const
    { return (ndims() == 3)
        ? 1 : cdesc_().weights_desc.dims[ndims() - (2 - with_groups())]; }
    inline int KW() const
    { return cdesc_().weights_desc.dims[ndims() - (1 - with_groups())]; }

    inline int KSD() const { return (ndims() == 5) ? cdesc_().strides[0] : 1; }
    inline int KSH() const { return (ndims() == 3)
        ? 1 : cdesc_().strides[ndims()-4]; }
    inline int KSW() const { return cdesc_().strides[ndims()-3]; }

    inline int KDD() const { return (ndims() == 5) ? cdesc_().dilates[0] : 0; }
    inline int KDH() const { return (ndims() == 3)
        ? 0 : cdesc_().dilates[ndims()-4]; }
    inline int KDW() const { return cdesc_().dilates[ndims()-3]; }

    inline int padFront() const
        { return (ndims() == 5) ? cdesc_().padding[0][0] : 0; }
    inline int padBack() const
        { return (ndims() == 5) ? cdesc_().padding[1][0] : 0; }
    inline int padT() const { return (ndims() == 3)
        ? 0 : cdesc_().padding[0][ndims()-4]; }
    inline int padB() const { return (ndims() == 3)
        ? 0 : cdesc_().padding[1][ndims()-4]; }
    inline int padL() const { return cdesc_().padding[0][ndims()-3]; }
    inline int padR() const { return cdesc_().padding[1][ndims()-3]; }

    inline float pad_value() const { return cdesc_().pad_value; }

    inline bool with_groups() const
    { return cdesc_().weights_desc.ndims == cdesc_().src_desc.ndims + 1; }

    inline int ndims() const { return cdesc_().src_desc.ndims; }

    bool has_zero_dim_memory() const {
        return false
            || memory_desc_wrapper(cdesc_().src_desc).has_zero_dim()
            || memory_desc_wrapper(cdesc_().dst_desc).has_zero_dim();
    }

protected:
    base_desc_t desc_;
    const _binary_convolution_fwd_pd_t *hint_fwd_pd_;

    inline const binary_convolution_desc_t &cdesc_() const;

    virtual status_t init() = 0;
};

using binary_convolution_fwd_pd_t = mkldnn::impl::_binary_convolution_fwd_pd_t;

inline const binary_convolution_desc_t &_binary_convolution_fwd_pd_t::cdesc_() const { return desc_; }

}
}

#endif
