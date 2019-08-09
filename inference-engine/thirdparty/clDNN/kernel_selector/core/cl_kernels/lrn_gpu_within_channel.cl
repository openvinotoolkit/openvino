// Copyright (c) 2016-2017 Intel Corporation
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


KERNEL (lrn_gpu_within_channel)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    for (uint index = get_global_id(0) ; index < INPUT0_LENGTH ; index += get_global_size(0))
    {
#if   defined OUTPUT_LAYOUT_YXFB
        const uint batch_id   = index % INPUT0_BATCH_NUM;
        const uint yxf        = index / INPUT0_BATCH_NUM;
        const uint feature_id = yxf   % INPUT0_FEATURE_NUM;
        const uint yx         = yxf   / INPUT0_FEATURE_NUM;
        const uint x          = yx    % INPUT0_SIZE_X;
        const uint y          = yx    / INPUT0_SIZE_X;
#elif defined OUTPUT_LAYOUT_BFYX
        const uint x          = index % INPUT0_SIZE_X;
        const uint bfy        = index / INPUT0_SIZE_X;
        const uint y          = bfy   % INPUT0_SIZE_Y;
        const uint bf         = bfy   / INPUT0_SIZE_Y;
        const uint feature_id = bf    % INPUT0_FEATURE_NUM;
        const uint batch_id   = bf    / INPUT0_FEATURE_NUM;
#endif

        const uint first_index_in_feature = INPUT0_OFFSET + batch_id*INPUT0_BATCH_PITCH + feature_id*INPUT0_FEATURE_PITCH;
        const uint input_id = first_index_in_feature + y*INPUT0_Y_PITCH + x*INPUT0_X_PITCH;

        int wstart = x - PADDING;
        int hstart = y - PADDING;
        int hend = min(hstart + LOCAL_SIZE, INPUT0_SIZE_Y + PADDING);
        int wend = min(wstart + LOCAL_SIZE, INPUT0_SIZE_X + PADDING);
        const int pool_size = (hend - hstart) * (wend - wstart);
        hstart = max(hstart, (int)0);
        wstart = max(wstart, (int)0);
        hend = min(hend, INPUT0_SIZE_Y);
        wend = min(wend, INPUT0_SIZE_X);
        UNIT_TYPE aveval = 0;

        __global const UNIT_TYPE* bottom_slice = input + first_index_in_feature;
        for (int h = hstart; h < hend; ++h)
        {
            for (int w = wstart; w < wend; ++w)
            {
                UNIT_TYPE tmp_val = bottom_slice[h*INPUT0_Y_PITCH + w*INPUT0_X_PITCH] * TO_UNIT_TYPE(ALPHA_VAL_FACTOR);
                aveval += (tmp_val * tmp_val);
            }
        }

        UNIT_TYPE acc = aveval / pool_size;
        acc = mad(acc, TO_UNIT_TYPE(ALPHA_AFTER_FACTORED), TO_UNIT_TYPE(K));
        acc = native_powr(acc, -TO_UNIT_TYPE(BETA));

        const uint output_idx = OUTPUT_OFFSET + batch_id*OUTPUT_BATCH_PITCH + feature_id*OUTPUT_FEATURE_PITCH + y*OUTPUT_Y_PITCH + x*OUTPUT_X_PITCH;
        output[output_idx] = ACTIVATION(acc * input[input_id], ACTIVATION_PARAMS);
    }
}