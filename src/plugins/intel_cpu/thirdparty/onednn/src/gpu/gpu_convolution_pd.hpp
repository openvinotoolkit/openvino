/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef GPU_GPU_CONVOLUTION_PD_HPP
#define GPU_GPU_CONVOLUTION_PD_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

struct gpu_convolution_fwd_pd_t : public convolution_fwd_pd_t {
    using convolution_fwd_pd_t::convolution_fwd_pd_t;

protected:
    bool zero_points_ok(const primitive_attr_t *attr) const {
        using namespace data_type;
        const auto src_type = invariant_src_md()->data_type;
        int mask_src = 0, mask_dst = 0;
        attr->zero_points_.get(DNNL_ARG_SRC, nullptr, &mask_src, nullptr);
        attr->zero_points_.get(DNNL_ARG_DST, nullptr, &mask_dst, nullptr);

        return IMPLICATION(!utils::one_of(src_type, s8, u8),
                       attr->zero_points_.has_default_values())
                && attr->zero_points_.has_default_values(DNNL_ARG_WEIGHTS)
                && (mask_src == 0 || mask_src == 1 << 1)
                && (mask_dst == 0 || mask_dst == 1 << 1);
    }
};

struct gpu_convolution_bwd_data_pd_t : public convolution_bwd_data_pd_t {
    using convolution_bwd_data_pd_t::convolution_bwd_data_pd_t;

protected:
    bool post_ops_ok(const primitive_attr_t *attr) const {
        const auto &p = attr->post_ops_;

        auto is_eltwise
                = [&](int idx) { return p.entry_[idx].is_eltwise(false); };
        auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(false); };

        switch (p.len()) {
            case 0: return true; // no post_ops
            case 1: return is_eltwise(0) || is_sum(0); // sum OR eltwise
            case 2: return is_sum(0) && is_eltwise(1); // sum -> eltwise
            default: return false;
        }

        return false;
    }
};

struct gpu_convolution_bwd_weights_pd_t : public convolution_bwd_weights_pd_t {
    using convolution_bwd_weights_pd_t::convolution_bwd_weights_pd_t;
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
