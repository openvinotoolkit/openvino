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

#include "include/include_all.cl"

#if INPUT0_LAYOUT_BFYX
#define IW INPUT0_SIZES[0]
#define IH INPUT0_SIZES[1]
#define IC INPUT0_SIZES[2]
#define IB INPUT0_SIZES[3]
#elif INPUT0_LAYOUT_BYXF
#define IC INPUT0_SIZES[0]
#define IW INPUT0_SIZES[1]
#define IH INPUT0_SIZES[2]
#define IB INPUT0_SIZES[3]
#endif

inline UNIT_TYPE FUNC(logistic_activate)(UNIT_TYPE x) {
    return 1. / (1. + exp(-x));
}

inline int FUNC(output_index)(int batch, int region_num, int xy, int feature_offset) {
    int region_offset = region_num * (COORDS + CLASSES + 1);

#if DO_SOFTMAX
    return OUTPUT_GET_INDEX(batch, (feature_offset + region_offset) * INPUT0_SIZE_X * INPUT0_SIZE_Y + xy, 1, 1);
#else
    int x_index = xy % INPUT0_SIZE_X;
    int y_index = (xy / INPUT0_SIZE_X) % (INPUT0_SIZE_Y);
    return OUTPUT_GET_INDEX(batch, feature_offset + region_offset, y_index, x_index);
#endif
}

KERNEL (region_yolo_ref)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    int xy = get_global_id(0);
    int region_num = get_global_id(1);
    int batch = get_global_id(2);
    int x_index = xy % INPUT0_SIZE_X;
    int y_index = (xy / INPUT0_SIZE_X) % (INPUT0_SIZE_Y);

    /// [x, y, width, height, objectness score, class score]
    /// x,y
    int region_offset = region_num * (COORDS + CLASSES + 1);
    int in_i = INPUT0_GET_INDEX(batch, 0 + region_offset, y_index, x_index);
    int out_i = FUNC_CALL(output_index)(batch, region_num, xy, 0);
    output[out_i] = FUNC_CALL(logistic_activate)(input[in_i]);

    in_i = INPUT0_GET_INDEX(batch, 1 + region_offset, y_index, x_index);
    out_i = FUNC_CALL(output_index)(batch, region_num, xy, 1);
    output[out_i] = FUNC_CALL(logistic_activate)(input[in_i]);

    /// width,height
    in_i = INPUT0_GET_INDEX(batch, 2 + region_offset, y_index, x_index);
    out_i = FUNC_CALL(output_index)(batch, region_num, xy, 2);
    output[out_i] = input[in_i];

    in_i = INPUT0_GET_INDEX(batch, 3 + region_offset, y_index, x_index);
    out_i = FUNC_CALL(output_index)(batch, region_num, xy, 3);
    output[out_i] = input[in_i];

    /// objectness score
    in_i = INPUT0_GET_INDEX(batch, COORDS + region_offset, y_index, x_index);
    out_i = FUNC_CALL(output_index)(batch, region_num, xy, COORDS);
    output[out_i] = FUNC_CALL(logistic_activate)(input[in_i]);

    /// class score(confidence)
#if DO_SOFTMAX
    in_i = INPUT0_GET_INDEX(batch, COORDS + 1 + region_offset, y_index, x_index);
    UNIT_TYPE max_value = input[in_i];
    for (int j = 1; j < CLASSES; j++) {
        in_i = INPUT0_GET_INDEX(batch, COORDS + 1 + j + region_offset, y_index, x_index);
        max_value = max(max_value, input[in_i]);
    }

    UNIT_TYPE expSum = 0;
    for (int j = 0; j < CLASSES; j++) {
        in_i = INPUT0_GET_INDEX(batch, COORDS + 1 + j + region_offset, y_index, x_index);
        out_i = FUNC_CALL(output_index)(batch, region_num, xy, COORDS + 1 + j);
        output[out_i] = exp(input[in_i] - max_value);
        expSum += output[out_i];
    }

    for (int j = 0; j < CLASSES; j++) {
        out_i = FUNC_CALL(output_index)(batch, region_num, xy, COORDS + 1 + j);
        output[out_i] /= expSum;
    }
#else
    for (int j = 0; j < CLASSES; j++)
    {
        in_i = INPUT0_GET_INDEX(batch, COORDS + 1 + j + region_offset, y_index, x_index);
        output[in_i] = FUNC_CALL(logistic_activate)(input[in_i]);
    }
#endif
}
