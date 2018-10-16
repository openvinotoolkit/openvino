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

#include <assert.h>
#include "mkldnn.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::alg_kind;

status_t mkldnn_convolution_relu_desc_init(
        convolution_relu_desc_t *conv_relu_desc,
        const convolution_desc_t *conv_desc, float negative_slope) {
    bool args_ok = !any_null(conv_relu_desc, conv_desc)
        && utils::one_of(conv_desc->prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    if (!args_ok) return invalid_arguments;
    conv_relu_desc->primitive_kind = primitive_kind::convolution_relu;
    conv_relu_desc->convolution_desc = *conv_desc;
    conv_relu_desc->negative_slope = negative_slope;
    return success;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
