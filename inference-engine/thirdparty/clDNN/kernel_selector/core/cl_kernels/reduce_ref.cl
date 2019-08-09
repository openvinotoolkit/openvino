// Copyright (c) 2019 Intel Corporation
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

#include "include/include_all.cl"

KERNEL(reduce_vec)(const __global INPUT0_TYPE* data, __global OUTPUT_TYPE* output) {
    const uint out_idx = get_global_id(0);

    if (out_idx >= COMPUTATIONAL_OPERATIONS_NUMBER) return;

#ifdef REDUCE_BATCH
    const uint batch_out = 0;
    const uint batch_max_val = INPUT0_BATCH_NUM;
#else
    const uint batch_out = BATCH_NUM_IDX_COMP(out_idx);
    const uint batch_max_val = batch_out + 1;
#endif

#ifdef REDUCE_FEATURE
    const uint feature_out = 0;
    const uint feature_max_val = INPUT0_FEATURE_NUM;
#else
    const uint feature_out = FEATURE_NUM_IDX_COMP(out_idx);
    const uint feature_max_val = feature_out + 1;
#endif

#if INPUT0_LAYOUT_BFWZYX
#ifdef REDUCE_W
    const uint w_out = 0;
    const uint w_max_val = INPUT0_SIZE_W;
#else
    const uint w_out = SIZE_W_IDX_COMP(out_idx);
    const uint w_max_val = w_out + 1;
#endif
#else
    const uint w_out = 0;
    const uint w_max_val = 1;
#endif

#if INPUT0_LAYOUT_BFWZYX || INPUT0_LAYOUT_BFZYX
#ifdef REDUCE_Z
    const uint z_out = 0;
    const uint z_max_val = INPUT0_SIZE_Z;
#else
    const uint z_out = SIZE_Z_IDX_COMP(out_idx);
    const uint z_max_val = z_out + 1;
#endif
#else
    const uint z_out = 0;
    const uint z_max_val = 1;
#endif

#ifdef REDUCE_Y
    const uint y_out = 0;
    const uint y_max_val = INPUT0_SIZE_Y;
#else
    const uint y_out = SIZE_Y_IDX_COMP(out_idx);
    const uint y_max_val = y_out + 1;
#endif

#ifdef REDUCE_X
    const uint x_out = 0;
    const uint x_max_val = INPUT0_SIZE_X;
#else
    const uint x_out = SIZE_X_IDX_COMP(out_idx);
    const uint x_max_val = x_out + 1;
#endif
    OUTPUT_TYPE acc = OUTPUT_VAL_ZERO;
    uint counter = 0;
    for (uint b = batch_out; b < batch_max_val; ++b) {
        for (uint f = feature_out; f < feature_max_val; ++f) {
            for (uint w = w_out; w < w_max_val; ++w) {
                for (uint z = z_out; z < z_max_val; ++z) {
                    for (uint y = y_out; y < y_max_val; ++y) {
                        for (uint x = x_out; x < x_max_val; ++x) {
#ifdef INPUT0_LAYOUT_BFWZYX
                            const uint input_idx = GET_DATA_INDEX_6D(INPUT0, b, f, w, z, y, x);
#elif INPUT0_LAYOUT_BFZYX
                            const uint input_idx = GET_DATA_INDEX_5D(INPUT0, b, f, z, y, x);
#else
                            const uint input_idx = GET_DATA_INDEX(INPUT0, b, f, y, x);
#endif
#ifdef REDUCE_SUM_MODE
                            acc += data[input_idx];
#elif REDUCE_MAX_MODE
                            if (counter == 0)
                                acc = data[input_idx];
                            else
                                acc = data[input_idx] > acc ? data[input_idx] : acc;
#elif REDUCE_MIN_MODE
                            if (counter == 0)
                                acc = data[input_idx];
                            else
                                acc = data[input_idx] < acc ? data[input_idx] : acc;
#elif REDUCE_MEAN_MODE
                            acc += data[input_idx];
#elif REDUCE_PROD_MODE
                            if (counter == 0)
                                acc = data[input_idx];
                            else
                                acc *= data[input_idx];
#elif REDUCE_AND_MODE
                            if (counter == 0)
                                acc = data[input_idx];
                            else
                                acc = acc && data[input_idx];
#elif REDUCE_OR_MODE
                            if (counter == 0)
                                acc = data[input_idx];
                            else
                                acc = acc || data[input_idx];
#elif REDUCE_SUM_SQUARE_MODE
                            acc += data[input_idx] * data[input_idx];
#elif REDUCE_L1_MODE
                            acc += fabs(data[input_idx]);
#elif REDUCE_L2_MODE
                            acc += data[input_idx] * data[input_idx];
#elif REDUCE_LOG_SUM_MODE
                            acc += data[input_idx];
#elif REDUCE_LOG_SUM_EXP_MODE
                            acc += exp(data[input_idx]);
#endif
                            counter++;
                        }
                    }
                }
            }
        }
    }
#if REDUCE_MEAN_MODE
    if (counter != 0) acc /= counter;
#endif
#if REDUCE_L2_MODE
    acc = sqrt(acc);
#endif
#if REDUCE_LOG_SUM_MODE || REDUCE_LOG_SUM_EXP_MODE
    acc = log(acc);
#endif

    output[out_idx] = acc;
}
