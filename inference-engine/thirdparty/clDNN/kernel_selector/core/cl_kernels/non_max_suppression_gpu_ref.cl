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
    __global OUTPUT_TYPE *output)
{
    int x = get_global_id(0);

#if NUM_SELECT_PER_CLASS_IDX
    printf("NUM_SELECT_PER_CLASS_IDX: %d\n", NUM_SELECT_PER_CLASS_IDX);
#endif

#if IOU_THRESHOLD_IDX
    printf("IOU_THRESHOLD_IDX: %d\n", IOU_THRESHOLD_IDX);
#endif

#if SCORE_THRESHOLD_IDX
    printf("SCORE_THRESHOLD_IDX: %d\n", SCORE_THRESHOLD_IDX);
#endif

#if SOFT_NMS_SIGMA_IDX
    printf("SOFT_NMS_SIGMA_IDX: %d\n", SOFT_NMS_SIGMA_IDX);
#endif

#if INPUT_ARG_SIZE
    printf("INPUT_ARG_SIZE: %d\n", INPUT_ARG_SIZE);
#endif
}
