/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include <assert.h>
#include <mkldnn_types.h>
#include "mkldnn.h"

#include "c_types_map.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::alg_kind;

namespace {
status_t roi_pooling_desc_init(roi_pooling_desc_t *roi_pool_desc,
        prop_kind_t prop_kind, alg_kind_t algorithm,
        memory_desc_t *src_descs, int num_src, const memory_desc_t *dst_desc,
        int pooled_h, int pooled_w, double spatial_scale) {
    
    roi_pooling_desc_t pd = {};
    pd.primitive_kind = primitive_kind::roi_pooling;
    pd.prop_kind = prop_kind;
    pd.pooled_h = pooled_h;
    pd.pooled_w = pooled_w;
    pd.spatial_scale = spatial_scale;
    pd.alg_kind = algorithm;

    pd.src_desc = src_descs;
    pd.num_src = num_src;
    pd.dst_desc = *dst_desc;

    *roi_pool_desc = pd;
    return success;
}
}

status_t mkldnn_roi_pooling_forward_desc_init(roi_pooling_desc_t *roi_pooling_desc,
        prop_kind_t prop_kind, alg_kind_t algorithm,
        memory_desc_t *src_desc, int num_src,
        const memory_desc_t *dst_desc,
        int pooled_h, int pooled_w, double spatial_scale) {
    if (!one_of(prop_kind, forward_inference))
        return invalid_arguments;

    if (!one_of(algorithm, mkldnn_roi_pooling_max, mkldnn_roi_pooling_bilinear))
        return invalid_arguments;

    return roi_pooling_desc_init(roi_pooling_desc, prop_kind, algorithm, src_desc, num_src, dst_desc,
        pooled_h, pooled_w, spatial_scale);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
