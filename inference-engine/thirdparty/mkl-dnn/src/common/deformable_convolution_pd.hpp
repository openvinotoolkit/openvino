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

#ifndef DEFORMABLE_CONVOLUTION_PD_HPP
#define DEFORMABLE_CONVOLUTION_PD_HPP
#include "mkldnn.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "memory_pd.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {

status_t def_conv_desc_init(deformable_convolution_desc_t *def_conv_desc,
                        prop_kind_t prop_kind, alg_kind_t alg_kind,
                        memory_desc_t *src_descs, int num_src, const memory_desc_t *weights_desc,
                        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc,
                        const dims_t strides, const dims_t dilates,
                        const dims_t padding_l, const dims_t padding_r,
                        padding_kind_t padding_kind, int deformable_group);

struct _deformable_convolution_fwd_pd_t: public primitive_desc_t {
    typedef _deformable_convolution_fwd_pd_t base_class;
    typedef _deformable_convolution_fwd_pd_t hint_class;
    static constexpr auto base_pkind = primitive_kind::deformable_convolution;

    _deformable_convolution_fwd_pd_t(mkldnn::impl::engine_t *engine,
                         const deformable_convolution_desc_t *adesc, const primitive_attr_t *attr,
                         const _deformable_convolution_fwd_pd_t *hint_fwd_pd)
            : primitive_desc_t(engine, attr, base_pkind), desc_(*adesc)
            , hint_fwd_pd_(hint_fwd_pd) {}
    virtual ~_deformable_convolution_fwd_pd_t() {}

    const deformable_convolution_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override
    { return reinterpret_cast<const op_desc_t *>(this->desc()); }
    virtual void init_info() override { init_info_def_conv(this, this->info_); }

    virtual const memory_pd_t *input_pd(int index = 0) const override {
        switch (index) {
            case 0: return src_pd(0);
            case 1: return src_pd(1);
            case 2: return weights_pd(0);
            case 3: return weights_pd(1);
            default: return nullptr;
        }
    }
    virtual const memory_pd_t *output_pd(int index = 0) const override
    { return index == 0 ? dst_pd() : nullptr; }

    virtual int n_inputs() const override { return 3 + with_bias(); }
    virtual int n_outputs() const override { return 1; }

    virtual status_t query(query_t what, int idx, void *result) const override
    {
        switch (what) {
            case pkind_traits<base_pkind>::query_d:
                *(const deformable_convolution_desc_t**)result = desc(); break;
            default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common conv aux functions */

    inline int MB() const { return input_pd()->desc()->dims[0]; }

    inline int IC() const { return input_pd()->desc()->dims[1]; }
    inline int OC() const { return output_pd()->desc()->dims[1]; }
    inline int G() const
    { return with_groups() ? desc_.weights_desc.dims[0] : 1; }

    inline int IH() const { return input_pd()->desc()->dims[ndims()-2]; }
    inline int IW() const { return input_pd()->desc()->dims[ndims()-1]; }
    inline int OH() const { return output_pd()->desc()->dims[ndims()-2]; }
    inline int OW() const { return output_pd()->desc()->dims[ndims()-1]; }
    inline int KH() const { return desc_.weights_desc.dims[ndims() - (2 - with_groups())]; }
    inline int KW() const { return desc_.weights_desc.dims[ndims() - (1 - with_groups())]; }

    inline int KSH() const { return desc_.strides[ndims()-4]; }
    inline int KSW() const { return desc_.strides[ndims()-3]; }

    inline int KDH() const { return desc_.dilates[ndims()-4]; }
    inline int KDW() const { return desc_.dilates[ndims()-3]; }

    inline int padT() const { return desc_.padding[0][ndims()-4]; }
    inline int padB() const { return desc_.padding[1][ndims()-4]; }
    inline int padL() const { return desc_.padding[0][ndims()-3]; }
    inline int padR() const { return desc_.padding[1][ndims()-3]; }

    inline int defGroup() const { return desc_.deformable_group; }

    inline bool with_bias() const
    { return !memory_desc_wrapper(desc_.bias_desc).is_zero(); }
    inline bool with_groups() const
    { return desc_.weights_desc.ndims == desc_.src_descs[0].ndims + 1; }

    inline int ndims() const { return desc_.src_descs[0].ndims; }

    virtual status_t set_alg_kind(alg_kind_t alg) {
        if (alg == alg_kind::undef) return status::invalid_arguments;
        desc_.alg_kind = alg;
        return status::success;
    }

    bool has_zero_dim_memory() const {
        return false
               || memory_desc_wrapper(desc_.src_descs[0]).has_zero_dim()
               || memory_desc_wrapper(desc_.src_descs[1]).has_zero_dim()
               || memory_desc_wrapper(desc_.dst_desc).has_zero_dim();
    }


protected:
    deformable_convolution_desc_t desc_;
    const _deformable_convolution_fwd_pd_t *hint_fwd_pd_;

    virtual status_t init() = 0;
};

using deformable_convolution_fwd_pd_t = mkldnn::impl::_deformable_convolution_fwd_pd_t;

}
}

#endif
