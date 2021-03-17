// Copyright (c) 2021 Intel Corporation
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

#include "include/data_types.cl"
#include "include/fetch.cl"

#if SORT_RESULT_DESCENDING
#endif

#if BOX_ENCODING
#endif

KERNEL (non_max_suppression_ref)(
    const __global INPUT0_TYPE *boxes,
    const __global INPUT1_TYPE *scores,
    __global OUTPUT_TYPE *output
    #if NUM_SELECT_PER_CLASS_IDX
    , const __global INPUT2_TYPE *num_select_per_class
    #endif
    #if IOU_THRESHOLD_IDX
    , const __global INPUT3_TYPE *iou_threshold
    #endif
    #if SCORE_THRESHOLD_IDX
    , const __global INPUT4_TYPE *score_threshold
    #endif
    #if SOFT_NMS_SIGMA_IDX
    , const __global INPUT5_TYPE *soft_nms_sigma
    #endif
    )
{
    int x = get_global_id(0);

    if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0)
    {
        for(int i = 0; i < 24; i++)
        {
            printf("%f ", boxes[i]);
        }
        printf("\n");

        for(int i = 0; i < 12; i++)
        {
            printf("%f ", scores[i]);
        }
        printf("\n");

#if NUM_SELECT_PER_CLASS_IDX
        printf("NUM_SELECT_PER_CLASS_IDX: %d, %f\n", NUM_SELECT_PER_CLASS_IDX, num_select_per_class[0]);
#endif

#if IOU_THRESHOLD_IDX
        printf("IOU_THRESHOLD_IDX: %d, %f\n", IOU_THRESHOLD_IDX, iou_threshold[0]);
#endif

#if SCORE_THRESHOLD_IDX
        printf("SCORE_THRESHOLD_IDX: %d, %f\n", SCORE_THRESHOLD_IDX, score_threshold[0]);
#endif

#if SOFT_NMS_SIGMA_IDX
        printf("SOFT_NMS_SIGMA_IDX: %d, %f\n", SOFT_NMS_SIGMA_IDX, soft_nms_sigma[0]);
#endif

#if INPUT_ARG_SIZE
        printf("INPUT_ARG_SIZE: %d\n", INPUT_ARG_SIZE);
#endif
    }
}
