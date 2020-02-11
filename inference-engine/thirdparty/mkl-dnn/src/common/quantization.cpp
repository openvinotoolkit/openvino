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
status_t quantization_desc_init(quantization_desc_t *quantization_desc, prop_kind_t prop_kind,
        alg_kind_t alg_kind, int axis,
        const memory_desc_t *src_desc,
        const memory_desc_t *thresholds_desc, const memory_desc_t *output_mask_desc,
        const memory_desc_t *crop_low_desc, const memory_desc_t *crop_high_desc,
        const memory_desc_t *input_scale_desc, const memory_desc_t *input_shift_desc,
        const memory_desc_t *output_scale_desc, const memory_desc_t *output_shift_desc,
        const memory_desc_t *dst_desc) {
    bool args_ok = true
        && !any_null(quantization_desc, src_desc, dst_desc)
        && one_of(prop_kind, forward_training, forward_inference)
        && one_of(alg_kind, binarization_depthwise, quantization_quantize_dequantize, quantization_quantize);
    if (!args_ok) return invalid_arguments;

    auto bd = quantization_desc_t();

    bd.axis = axis;

    if (alg_kind == binarization_depthwise) {
        if (any_null(quantization_desc, thresholds_desc, output_mask_desc))
            return invalid_arguments;

        bd.thresholds_desc = *thresholds_desc;
        bd.output_mask_desc = *output_mask_desc;

        bd.crop_low_desc = zero_md();
        bd.crop_high_desc = zero_md();
        bd.input_scale_desc = zero_md();
        bd.input_shift_desc = zero_md();
        bd.output_scale_desc = zero_md();
        bd.output_shift_desc = zero_md();
    } else {
        if (any_null(quantization_desc, crop_low_desc, crop_high_desc, input_scale_desc, input_shift_desc,
                output_scale_desc, output_shift_desc))
            return invalid_arguments;

        bd.thresholds_desc = zero_md();
        bd.output_mask_desc = zero_md();

        bd.crop_low_desc = *crop_low_desc;
        bd.crop_high_desc = *crop_high_desc;
        bd.input_scale_desc = *input_scale_desc;
        bd.input_shift_desc = *input_shift_desc;
        bd.output_scale_desc = *output_scale_desc;
        bd.output_shift_desc = *output_shift_desc;
    }

    bd.primitive_kind = primitive_kind::quantization;
    bd.prop_kind = prop_kind;
    bd.alg_kind = alg_kind;
    bd.src_desc = *src_desc;
    bd.dst_desc = *dst_desc;

    bool consistency = true
        && memory_desc_wrapper(bd.src_desc).nelems()
        && memory_desc_wrapper(bd.dst_desc).nelems();
    if (!consistency) return invalid_arguments;

    *quantization_desc = bd;
    return success;
}
}

status_t mkldnn_binarization_forward_desc_init(quantization_desc_t *quantization_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind, int axis,
        const memory_desc_t *src_desc, const memory_desc_t *thresholds_desc, const memory_desc_t *output_mask_desc, const memory_desc_t *dst_desc) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;
    return quantization_desc_init(quantization_desc, prop_kind, alg_kind, axis, src_desc, thresholds_desc, output_mask_desc,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, dst_desc);
}

status_t mkldnn_quantization_forward_desc_init(quantization_desc_t *quantization_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind, int axis,
        const memory_desc_t *src_desc,
        const memory_desc_t *crop_low_desc, const memory_desc_t *crop_high_desc, const memory_desc_t *input_scale_desc,
        const memory_desc_t *input_shift_desc, const memory_desc_t *output_scale_desc, const memory_desc_t *output_shift_desc,
        const memory_desc_t *dst_desc) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;
    return quantization_desc_init(quantization_desc, prop_kind, alg_kind, axis, src_desc, nullptr, nullptr, crop_low_desc, crop_high_desc,
            input_scale_desc, input_shift_desc, output_scale_desc, output_shift_desc, dst_desc);
}