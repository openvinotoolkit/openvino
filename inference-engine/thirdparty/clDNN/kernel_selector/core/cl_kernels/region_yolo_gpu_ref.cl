// Copyright (c) 2017 Intel Corporation
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

#define IW INPUT0_SIZES[0]
#define IH INPUT0_SIZES[1]
#define IC INPUT0_SIZES[2]
#define IB INPUT0_SIZES[3]

inline UNIT_TYPE FUNC(logistic_activate)(UNIT_TYPE x) {
    return 1. / (1. + exp(-x));
}

inline int FUNC(entry_index)(int width, int height, int coords, int classes,
                       int outputs, int batch, int location,
                       int entry) {
    int n = location / (width * height);
    int loc = location % (width * height);
    return batch * outputs + n * width * height * (coords + classes + 1) +
        entry * width * height + loc;
}

#if DO_SOFTMAX
inline void FUNC(softmax_generic)(const __global UNIT_TYPE* src_data, __global UNIT_TYPE* dst_data,
                            int B, int C, int W, int H, int i)
{
    for (int b = 0; b < B; b++) {
        UNIT_TYPE max = src_data[b*C*H*W + i];
        for (int c = 0; c < C; c++) {
            UNIT_TYPE val = src_data[b*C*H*W + c*H*W + i];
            if (val > max) max = val;
        }

        UNIT_TYPE expSum = 0;
        for (int c = 0; c < C; c++) {
            dst_data[b*C*H*W + c*H*W + i] = exp(src_data[b*C*H*W + c*H*W + i] - max);
            expSum += dst_data[b*C*H*W + c*H*W + i];
        }

        for (int c = 0; c < C; c++) {
            dst_data[b*C*H*W + c*H*W + i] = dst_data[b*C*H*W + c*H*W + i] / expSum;
        }
    }
}
#endif

KERNEL (region_yolo_ref)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    int x = get_global_id(0);

#if DO_SOFTMAX
    #define ACTUAL_NUM (NUM)
    #define CONF_CLASSES (1)
#else
    #define ACTUAL_NUM (MASK_SIZE)
    #define CONF_CLASSES (CLASSES+1)
#endif
    #define INPUTS_COUNT (IH * IW * ACTUAL_NUM * (CLASSES + COORDS + 1))

    for (int b = 0; b < IB; b++) {
        for (int n = 0; n < ACTUAL_NUM; n++) {
            // coords: x/y
            int index = FUNC_CALL(entry_index)(IW, IH, COORDS, CLASSES, INPUTS_COUNT, b, n * IW * IH, 0);
            int i = index + 2 * x;
            output[i] = FUNC_CALL(logistic_activate)(input[i]);
            output[i+1] = FUNC_CALL(logistic_activate)(input[i+1]);

            // coords: w/h: directly copy?
            index = FUNC_CALL(entry_index)(IW, IH, COORDS, CLASSES, INPUTS_COUNT, b, n * IW * IH, 2);
            i = index + 2 * x;
            output[i] = input[i];
            output[i+1] = input[i+1];

            // confidence
            index = FUNC_CALL(entry_index)(IW, IH, COORDS, CLASSES, INPUTS_COUNT, b, n * IW * IH, COORDS);
            for (int j = 0; j < CONF_CLASSES; j++)
            {
                i = index + x + j*IH*IW;
                output[i] = FUNC_CALL(logistic_activate)(input[i]);
            }
        }
    }

#if DO_SOFTMAX
    // the probability of classes
    int index = FUNC_CALL(entry_index)(IW, IH, COORDS, CLASSES, INPUTS_COUNT, 0, 0, COORDS + 1);
    int batch_offset = INPUTS_COUNT / NUM;
    for (int b = 0; b < IB * NUM; b++)
        FUNC_CALL(softmax_generic)(input + index + b * batch_offset, output + index + b * batch_offset,
                                   1, CLASSES, IH, IW, x);
#endif
}
