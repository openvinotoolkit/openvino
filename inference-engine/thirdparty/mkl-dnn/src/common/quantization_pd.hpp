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

#ifndef QUANTIZATION_PD_HPP
#define QUANTIZATION_PD_HPP

#include <mkldnn_types.h>
#include "mkldnn.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "memory_pd.hpp"

namespace mkldnn {
namespace impl {

struct quantization_fwd_pd_t: public primitive_desc_t {
    typedef quantization_fwd_pd_t base_class;
    typedef quantization_fwd_pd_t hint_class;
    static constexpr auto base_pkind = primitive_kind::quantization;

    quantization_fwd_pd_t(mkldnn::impl::engine_t *engine,
            const quantization_desc_t *adesc, const primitive_attr_t *attr,
            const quantization_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, attr, primitive_kind::quantization)
        , desc_(*adesc), hint_fwd_pd_(hint_fwd_pd) {}
    virtual ~quantization_fwd_pd_t() {}

    const quantization_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override
    { return reinterpret_cast<const op_desc_t *>(this->desc()); }
    virtual void init_info() override { init_info_quantization(this, this->info_); }

    virtual const memory_pd_t *input_pd(int index = 0) const override {
        switch (index) {
        case 0:
            return src_pd();
        case 1: case 2:
            return weights_pd(index - 1);
        case 3: case 4: case 5: case 6:
            if (!is_binarization())
                return weights_pd(index - 1);
            else
                return nullptr;
        default: return nullptr;
        }
    }
    virtual const memory_pd_t *output_pd(int index = 0) const override
    { return index == 0 ? dst_pd() : nullptr; }

    virtual int n_inputs() const override { return is_binarization() ? 3 : 7; }
    virtual int n_outputs() const override { return 1; }

    virtual status_t query(query_t what, int idx, void *result) const override
    {
        switch (what) {
        case query::quantization_d:
            *(const quantization_desc_t**)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common quantization aux functions */

    inline bool is_binarization() const { return desc_.alg_kind == alg_kind::binarization_depthwise; }

    inline int MB() const { return input_pd()->desc()->ndims > 0 ? input_pd()->desc()->dims[0] : 1; }
    inline int C()  const { return input_pd()->desc()->ndims > 1 ? input_pd()->desc()->dims[1] : 1; }
    inline int D()  const { return input_pd()->desc()->ndims > 4 ? input_pd()->desc()->dims[2] : 1; }
    inline int H()  const { return input_pd()->desc()->ndims > 4 ? input_pd()->desc()->dims[3] :
                                   input_pd()->desc()->ndims > 2 ? input_pd()->desc()->dims[2] : 1; }
    inline int W()  const { return input_pd()->desc()->ndims > 4 ? input_pd()->desc()->dims[4] :
                                   input_pd()->desc()->ndims > 3 ? input_pd()->desc()->dims[3] : 1; }

    inline int axis() const { return desc()->axis; }

protected:
    quantization_desc_t desc_;
    const quantization_fwd_pd_t *hint_fwd_pd_;
};

}
}

#endif

