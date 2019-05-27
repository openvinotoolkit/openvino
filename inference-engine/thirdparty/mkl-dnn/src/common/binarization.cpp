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

#include <assert.h>
#include <mkldnn_types.h>
#include "mkldnn.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::alg_kind;
using namespace mkldnn::impl::types;

namespace {
status_t binarization_desc_init(binarization_desc_t *binarization_desc, prop_kind_t prop_kind,
        alg_kind_t alg_kind, const memory_desc_t *src_desc, const memory_desc_t *dst_desc,
        const memory_desc_t *weights_desc, const memory_desc_t *output_mask_desc) {
    bool args_ok = true
        && !any_null(binarization_desc, src_desc, dst_desc, weights_desc, output_mask_desc)
        && one_of(prop_kind, forward_training, forward_inference)
        && one_of(alg_kind, binarization_depthwise);
    if (!args_ok) return invalid_arguments;

    auto bd = binarization_desc_t();
    bd.primitive_kind = primitive_kind::binarization;
    bd.prop_kind = prop_kind;
    bd.alg_kind = alg_kind;
    bd.src_desc = *src_desc;
    bd.dst_desc = *dst_desc;
    bd.weights_desc = *weights_desc;
    bd.output_mask_desc = *output_mask_desc;

    bool consistency = true
        && memory_desc_wrapper(bd.src_desc).nelems()
        && memory_desc_wrapper(bd.dst_desc).nelems();
    if (!consistency) return invalid_arguments;

    *binarization_desc = bd;
    return success;
}
}

status_t mkldnn_binarization_forward_desc_init(binarization_desc_t *binarization_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *dst_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *output_mask_desc) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;
    return binarization_desc_init(binarization_desc, prop_kind, alg_kind, src_desc, dst_desc, weights_desc, output_mask_desc);
}
