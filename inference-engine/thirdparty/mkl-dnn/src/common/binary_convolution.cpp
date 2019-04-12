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

namespace mkldnn {
namespace impl {
status_t bin_conv_desc_init(binary_convolution_desc_t *bin_conv_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *dst_desc,
        const dims_t strides, const dims_t dilates,
        const dims_t padding_l, const dims_t padding_r,
        float pad_value) {
    bool args_ok = true
        && !any_null(bin_conv_desc, src_desc, weights_desc, dst_desc, strides,
                padding_l)
        && one_of(alg_kind, binary_convolution_direct)
        && one_of(pad_value, -1.f, 0.f, 1.f);
    if (!args_ok) return invalid_arguments;

    if (padding_r == nullptr) padding_r = padding_l;

    auto bcd = binary_convolution_desc_t();
    bcd.primitive_kind = primitive_kind::binary_convolution;
    bcd.prop_kind = prop_kind;
    bcd.alg_kind = alg_kind;

    bcd.src_desc = zero_md();
    bcd.dst_desc = zero_md();
    bcd.weights_desc = zero_md();

    const bool with_groups = weights_desc->ndims == src_desc->ndims + 1;

    bcd.src_desc = *src_desc;
    bcd.dst_desc = *dst_desc;
    bcd.weights_desc = *weights_desc;

    int sp_dims = src_desc->ndims - 2;
    utils::array_copy(bcd.strides, strides, sp_dims);
    utils::array_copy(bcd.padding[0], padding_l, sp_dims);
    utils::array_copy(bcd.padding[1], padding_r, sp_dims);
    if (dilates)
        utils::array_copy(bcd.dilates, dilates, sp_dims);
    else
        utils::array_set(bcd.dilates, 0, sp_dims);

    bcd.pad_value = pad_value;
    bcd.accum_data_type = types::default_accum_data_type(src_desc->data_type,
            weights_desc->data_type, dst_desc->data_type, prop_kind);

    bool consistency = true
        && memory_desc_wrapper(weights_desc).nelems()
        && src_desc->ndims == dst_desc->ndims
        && utils::one_of(src_desc->ndims, 3, 4, 5)
        && utils::one_of(weights_desc->ndims, src_desc->ndims, src_desc->ndims + 1)
        && src_desc->dims[0] == dst_desc->dims[0];
    for (int i = 2; i < src_desc->ndims; ++i)
    {
        int src = src_desc->dims[i];
        int ker = weights_desc->dims[with_groups + i];
        int dil = bcd.dilates[i - 2];
        int pad_l = padding_l[i - 2];
        int pad_r = padding_r[i - 2];
        int str = strides[i - 2];
        int dst = dst_desc->dims[i];
        int ker_range = 1 + (ker - 1) * (dil + 1);

        if (str < 1) return invalid_arguments;
        consistency = consistency
            && dil >= 0
            && pad_l >= 0
//            && pad_r + str > 0 // TODO: [dmitrygo] Commented as WA to support dw conv fusing
            && (src - ker_range + pad_l + pad_r) / str + 1 == dst;
    }
    if (!consistency) return invalid_arguments;

    *bin_conv_desc = bcd;
    return success;
}
}
}

status_t mkldnn_dilated_binary_convolution_forward_desc_init(
        binary_convolution_desc_t *bin_conv_desc, prop_kind_t prop_kind,
        alg_kind_t alg_kind, const memory_desc_t *src_desc,
        const memory_desc_t *weights_desc, const memory_desc_t *dst_desc, const dims_t strides,
        const dims_t dilates, const dims_t padding_l,
        const dims_t padding_r,
        const float pad_value) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;
    return mkldnn::impl::bin_conv_desc_init(bin_conv_desc, prop_kind, alg_kind, src_desc,
            weights_desc, dst_desc, strides, dilates, padding_l, padding_r, pad_value);
}
