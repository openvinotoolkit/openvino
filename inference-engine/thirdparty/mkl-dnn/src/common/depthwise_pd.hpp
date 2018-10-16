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

#ifndef DEPTHWISE_PD_HPP
#define DEPTHWISE_PD_HPP

#include <mkldnn_types.h>
#include "mkldnn.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "memory_pd.hpp"

namespace mkldnn {
namespace impl {

struct depthwise_fwd_pd_t: public primitive_desc_t {
    typedef depthwise_fwd_pd_t base_class;
    typedef depthwise_fwd_pd_t hint_class;
    static constexpr auto base_pkind = primitive_kind::depthwise;

    depthwise_fwd_pd_t(mkldnn::impl::engine_t *engine,
            const depthwise_desc_t *adesc, const primitive_attr_t *attr,
            const depthwise_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, attr, primitive_kind::depthwise)
        , desc_(*adesc), hint_fwd_pd_(hint_fwd_pd) {}
    virtual ~depthwise_fwd_pd_t() {}

    const depthwise_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override
    { return reinterpret_cast<const op_desc_t *>(this->desc()); }
    virtual void init_info() override { init_info_depthwise(this, this->info_); }

    virtual const memory_pd_t *input_pd(int index = 0) const override {
        switch (index) {
        case 0: return src_pd();
        case 1: case 2: return weights_pd(index - 1);
        default: return nullptr;
        }
    }
    virtual const memory_pd_t *output_pd(int index = 0) const override
    { return index == 0 ? dst_pd() : nullptr; }

    virtual int n_inputs() const override { return 2 + with_bias(); }
    virtual int n_outputs() const override { return 1; }

    virtual status_t query(query_t what, int idx, void *result) const override
    {
        switch (what) {
        case query::depthwise_d:
            *(const depthwise_desc_t**)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    inline bool with_bias() const
    { return !memory_desc_wrapper(this->desc()->bias_desc).is_zero(); }

    /* common depthwise aux functions */

    inline int MB() const { return input_pd()->desc()->ndims > 0 ? input_pd()->desc()->dims[0] : 1; }
    inline int C()  const { return input_pd()->desc()->ndims > 1 ? input_pd()->desc()->dims[1] : 1; }
    inline int H()  const { return input_pd()->desc()->ndims > 2 ? input_pd()->desc()->dims[2] : 1; }
    inline int W()  const { return input_pd()->desc()->ndims > 3 ? input_pd()->desc()->dims[3] : 1; }

protected:
    depthwise_desc_t desc_;
    const depthwise_fwd_pd_t *hint_fwd_pd_;
};

}
}

#endif

