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

status_t def_conv_desc_init(deformable_convolution_desc_t *def_conv_desc,
                        prop_kind_t prop_kind, alg_kind_t alg_kind,
                        memory_desc_t *src_descs, int num_src, const memory_desc_t *weights_desc,
                        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc,
                        const dims_t strides, const dims_t dilates,
                        const dims_t padding_l, const dims_t padding_r,
                        padding_kind_t padding_kind, const int deformable_group) {
    bool args_ok = true
                   && !any_null(def_conv_desc, src_descs, weights_desc, dst_desc, strides, dilates, padding_l)
                   && one_of(alg_kind, deformable_convolution_direct)
                   && one_of(padding_kind, padding_kind::padding_zero);
    if (!args_ok) return invalid_arguments;

    if (padding_r == nullptr) padding_r = padding_l;

    auto cd = deformable_convolution_desc_t();
    cd.primitive_kind = primitive_kind::deformable_convolution;
    cd.prop_kind = prop_kind;
    cd.alg_kind = alg_kind;

    cd.dst_desc = zero_md();
    cd.weights_desc = zero_md();
    cd.bias_desc = zero_md();

    const bool with_bias = bias_desc && bias_desc->format != memory_format::undef;
    const bool with_groups = weights_desc->ndims == src_descs[0].ndims + 1;

    cd.src_descs = src_descs;
    cd.num_src = num_src;
    cd.dst_desc = *dst_desc;
    cd.weights_desc = *weights_desc;
    if (with_bias)
        cd.bias_desc = *bias_desc;

    int sp_dims = src_descs[0].ndims - 2;
    utils::array_copy(cd.strides, strides, sp_dims);
    utils::array_copy(cd.dilates, dilates, sp_dims);
    utils::array_copy(cd.padding[0], padding_l, sp_dims);
    utils::array_copy(cd.padding[1], padding_r, sp_dims);
    if (dilates)
        utils::array_copy(cd.dilates, dilates, sp_dims);
    else
        utils::array_set(cd.dilates, 0, sp_dims);

    cd.deformable_group = deformable_group;
    cd.padding_kind = padding_kind;
    cd.accum_data_type = types::default_accum_data_type(src_descs[0].data_type,
                                                        weights_desc->data_type, dst_desc->data_type, prop_kind);

    const int g = with_groups ? weights_desc->dims[0] : 1;
    const int bias_dim = dst_desc->dims[1];

    bool consistency = true
                       && memory_desc_wrapper(weights_desc).nelems()
                       && src_descs[0].ndims == src_descs[1].ndims
                       && src_descs[0].ndims == dst_desc->ndims
                       && utils::one_of(src_descs[0].ndims, 4)
                       && utils::one_of(weights_desc->ndims, src_descs[0].ndims,
                                        src_descs[0].ndims + 1)
                       && (with_bias ? bias_desc->ndims == 1 : true)
                       && (with_bias ? bias_desc->dims[0] == bias_dim : true)
                       && src_descs[0].dims[0] == dst_desc->dims[0]
                       && src_descs[0].dims[1] == g * weights_desc->dims[with_groups + 1]
                       && dst_desc->dims[1] == g * weights_desc->dims[with_groups + 0]
                       && src_descs[1].dims[1] == deformable_group * sp_dims * weights_desc->dims[with_groups + 2] * weights_desc->dims[with_groups + 3];

    for (int i = 2; i < src_descs[0].ndims; ++i)
    {
        int src = src_descs[0].dims[i];
        int ker = weights_desc->dims[with_groups + i];
        int dil = cd.dilates[i - 2];
        int pad_l = padding_l[i - 2];
        int pad_r = padding_r[i - 2];
        int str = strides[i - 2];
        int dst = dst_desc->dims[i];
        int ker_range = 1 + (ker - 1) * (dil + 1);

        if (str < 1) return invalid_arguments;
        consistency = consistency
                      && dil >= 0
                      && pad_l >= 0
                      && (src - ker_range + pad_l + pad_r) / str + 1 == dst;
    }
    if (!consistency) return invalid_arguments;

    *def_conv_desc = cd;
    return success;
}

}
}

status_t mkldnn_deformable_convolution_forward_desc_init(deformable_convolution_desc_t *def_conv_desc,
                                              prop_kind_t prop_kind, alg_kind_t alg_kind,
                                              memory_desc_t *src_descs, int num_src, const memory_desc_t *weights_desc,
                                              const memory_desc_t *bias_desc, const memory_desc_t *dst_desc,
                                              const dims_t strides, const dims_t dilates, const dims_t padding_l, const dims_t padding_r,
                                              padding_kind_t padding_kind, const int deformable_group) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;
    return mkldnn::impl::def_conv_desc_init(def_conv_desc, prop_kind, alg_kind, src_descs, num_src,
                                        weights_desc, bias_desc, dst_desc, strides, dilates,
                                        padding_l, padding_r, padding_kind, deformable_group);
}
