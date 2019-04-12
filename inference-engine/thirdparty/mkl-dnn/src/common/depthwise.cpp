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
status_t depthwise_desc_init(depthwise_desc_t *depthwise_desc, prop_kind_t prop_kind,
        alg_kind_t alg_kind, const memory_desc_t *src_desc, const memory_desc_t *dst_desc,
        const memory_desc_t *weights_desc, const memory_desc_t *bias_desc) {
    bool args_ok = true
        && !any_null(depthwise_desc, src_desc, dst_desc)
        && one_of(prop_kind, forward_training, forward_inference)
        && one_of(alg_kind, depthwise_scale_shift, depthwise_prelu);
    if (!args_ok) return invalid_arguments;

    auto dd = depthwise_desc_t();
    dd.primitive_kind = primitive_kind::depthwise;
    dd.prop_kind = prop_kind;
    dd.alg_kind = alg_kind;
    dd.src_desc = *src_desc;
    dd.dst_desc = *dst_desc;
    dd.weights_desc = *weights_desc;

    const bool with_bias = bias_desc && bias_desc->format != memory_format::undef;
    dd.bias_desc = with_bias ? *bias_desc : zero_md();

    bool consistency = true
        && memory_desc_wrapper(dd.src_desc).nelems()
        && memory_desc_wrapper(dd.dst_desc).nelems();
    if (!consistency) return invalid_arguments;

    *depthwise_desc = dd;
    return success;
}
}

status_t mkldnn_depthwise_forward_desc_init(depthwise_desc_t *depthwise_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *dst_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;
    return depthwise_desc_init(depthwise_desc, prop_kind, alg_kind, src_desc, dst_desc,
                               weights_desc, bias_desc);
}

status_t mkldnn_depthwise_forward_desc_init(depthwise_desc_t *depthwise_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *dst_desc, const memory_desc_t *weights_desc) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;
    return depthwise_desc_init(depthwise_desc, prop_kind, alg_kind, src_desc, dst_desc,
                               weights_desc, nullptr);
}
