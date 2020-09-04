// Copyright (c) 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include/common.cl"
#include "include/data_types.cl"
#include "include/fetch.cl"

__attribute__((intel_reqd_sub_group_size(16)))
KERNEL (lrn_gpu_across_channel_multiple_features_fsv16)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const uint feature_id   = (uint)get_global_id(0);
    const uint x            = (uint)get_global_id(1);
    const uint b_y          = (uint)get_global_id(2);
    const uint batch_id     = b_y / INPUT0_SIZE_Y;
    const uint y            = b_y % INPUT0_SIZE_Y;

    if (feature_id >= INPUT0_FEATURE_NUM)
        return;

    int input_offset_f = feature_id - PADDING;

    INPUT0_TYPE val[LOCAL_SIZE];
    INPUT0_TYPE res = 0;
    for (uint i = 0; i < LOCAL_SIZE; ++i, ++input_offset_f) {
        bool non_zero = input_offset_f >= 0 && input_offset_f < INPUT0_FEATURE_NUM;
        uint input_idx = INPUT0_GET_INDEX(batch_id, input_offset_f, y, x);
        val[i] = (int)non_zero * TO_INPUT0_TYPE(input[input_idx]);
        res = mad(val[i], val[i], res);
    }
    res = mad(res, TO_INPUT0_TYPE(ALPHA_DIV_BY_SIZE), TO_INPUT0_TYPE(K));
    res = native_powr(res, -TO_INPUT0_TYPE(BETA));

    uint output_idx = OUTPUT_GET_INDEX(batch_id, feature_id, y, x);
    INPUT0_TYPE lrn_result = res * val[PADDING];
    #if HAS_FUSED_OPS
        FUSED_OPS;
        output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT);
    #else
        output[output_idx] = ACTIVATION(TO_OUTPUT_TYPE(lrn_result), ACTIVATION_PARAMS);
    #endif
}
