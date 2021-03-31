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

/// optional input variables
/// NUM_SELECT_PER_CLASS_VAL TO_UNIT_TYPE(num_select_per_class[0]),   default is 0 
/// IOU_THRESHOLD_VAL        TO_ACCUMULATOR_TYPE(iou_threshold[0]),   default is ACCUMULATOR_VAL_ZERO
/// SCORE_THRESHOLD_VAL      TO_ACCUMULATOR_TYPE(score_threshold[0]), default is ACCUMULATOR_VAL_ZERO
/// SOFT_NMS_SIGMA_VAL       TO_ACCUMULATOR_TYPE(soft_nms_sigma[0]),  default is ACCUMULATOR_VAL_ZERO
/// SELECTED_SCORES_TERM     1 if selected_scores is used
/// VALID_OUTPUTS_TERM       1 if valid_outputs is used
/// OUTPUT_NUM               Number of outputs. [OUTPUT_NUM, 3, 1, 1]
/// BUFFER_STRIDE            20 bytes * NUM_BOXES
#if SORT_RESULT_DESCENDING
#endif

#if BOX_ENCODING
#endif

#define unroll_for __attribute__((opencl_unroll_hint)) for

#define NUM_BATCHES     INPUT0_BATCH_NUM
#define NUM_BOXES       INPUT0_FEATURE_NUM
#define NUM_CLASSES     INPUT1_FEATURE_NUM

typedef struct {
    int boxId;
    int suppress_begin_index;
    INPUT1_TYPE score;
} FUNC(SortedBoxInfo);

typedef struct {
    int batchId;
    int classId;
    int boxId;
    INPUT1_TYPE score;
} FUNC(BoxInfo);

#define SBOX_INFO FUNC(SortedBoxInfo)
#define BOX_INFO FUNC(BoxInfo)

inline float FUNC(intersectionOverUnion)(const __global INPUT0_TYPE *boxes,
    int batchA, int boxIdA,
    int batchB, int boxIdB)
{
    const float4 pA = convert_float4(vload4(0, &boxes[INPUT0_GET_INDEX(batchA, boxIdA, 0, 0)]));
    const float4 pB = convert_float4(vload4(0, &boxes[INPUT0_GET_INDEX(batchB, boxIdB, 0, 0)]));

    const float areaA = (pA[3] - pA[1]) * (pA[2] - pA[0]);
    const float areaB = (pB[3] - pB[1]) * (pB[2] - pB[0]);

    const float intersection_ymin = max(pA[1], pB[1]);
    const float intersection_xmin = max(pA[0], pB[0]);
    const float intersection_ymax = min(pA[3], pB[3]);
    const float intersection_xmax = min(pA[2], pB[2]);

    const float intersection_x = max(intersection_xmax - intersection_xmin, 0.f);
    const float intersection_y = max(intersection_ymax - intersection_ymin, 0.f);
    if (intersection_x == 0.f || intersection_y == 0.f)
        return 0.f;

    const float intersection_area = intersection_x * intersection_y;
    const float union_area = areaA + areaB - intersection_area;
    if (union_area <= 0.0f)
        return 0.f;

    return intersection_area / union_area;
}

inline float FUNC(scaleIOU)(float iou, float iou_threshold, float scale)
{
    //printf("iou(%.2f) iou_threshold(%.2f) scale(%.2f)\n", iou, iou_threshold, scale);
    if (iou <= iou_threshold) {
        return exp(scale * iou * iou);
    } else {
        return 0.0f;
    }
}

inline void FUNC(swap_sbox_info)(__global SBOX_INFO* a, __global SBOX_INFO* b)
{
    SBOX_INFO temp = *a;
    *a = *b;
    *b = temp;
}

inline int FUNC(partition)(__global SBOX_INFO* arr, int l, int h)
{
    INPUT1_TYPE x = arr[h].score;
    int i = (l - 1);

    for (int j = l; j <= h - 1; j++) {
        if (arr[j].score <= x) {
            i++;
            FUNC_CALL(swap_sbox_info)(&arr[i], &arr[j]);
        }
    }
    FUNC_CALL(swap_sbox_info)(&arr[i + 1], &arr[h]);
    return (i + 1);
}


inline void FUNC(quickSortIterative)(__global SBOX_INFO* arr, int l, int h)
{
    // Create an auxiliary stack
    int stack[NUM_BOXES];

    // initialize top of stack
    int top = -1;

    // push initial values of l and h to stack
    stack[++top] = l;
    stack[++top] = h;

    // Keep popping from stack while is not empty
    while (top >= 0) {
        // Pop h and l
        h = stack[top--];
        l = stack[top--];
  
        // Set pivot element at its correct position
        // in sorted array
        int p = FUNC_CALL(partition)(arr, l, h);
  
        // If there are elements on left side of pivot,
        // then push left side to stack
        if (p - 1 > l) {
            stack[++top] = l;
            stack[++top] = p - 1;
        }
  
        // If there are elements on right side of pivot,
        // then push right side to stack
        if (p + 1 < h) {
            stack[++top] = p + 1;
            stack[++top] = h;
        }
    }
}

inline int FUNC(initBoxList)(__global SBOX_INFO *outBoxes, int boxNum, const __global INPUT1_TYPE *scores, float score_threshold, int batchId, int classId)
{
    int count = 0;
    for (int i = 0; i < boxNum; ++i) {
        INPUT1_TYPE score = scores[INPUT1_GET_INDEX(batchId, classId, i, 0)];
        if (convert_float(score) < score_threshold) continue;

        SBOX_INFO binfo;
        binfo.boxId = i;
        binfo.suppress_begin_index = 0;
        binfo.score = score;
        outBoxes[count] = binfo;
        ++count;
    }

    return count;
}

inline void FUNC(initOutputBoxList)(__global BOX_INFO *outBoxes, int boxNum, const __global INPUT1_TYPE *scores, __global OUTPUT_TYPE *output)
{
    for (int i = 0; i < boxNum; ++i) {
        int outputId = i * 3;
        outBoxes[i].batchId = output[outputId + 0];
        outBoxes[i].classId = output[outputId + 1];
        outBoxes[i].boxId = output[outputId + 2];
        outBoxes[i].score = scores[INPUT1_GET_INDEX(outBoxes[i].batchId, outBoxes[i].classId, outBoxes[i].boxId, 0)];
    }
}

inline void FUNC(swap)(__global BOX_INFO* a, __global BOX_INFO* b)
{
    BOX_INFO temp = *a;
    *a = *b;
    *b = temp;
}

inline void FUNC(sortOutputBoxList)(__global BOX_INFO *outSortedBoxes, int boxNum)
{
    for (int i = 0; i < boxNum - 1; ++i) {
        bool swapped = false;
        for (int j = 0; j < boxNum - i - 1; ++j) {
            if ((outSortedBoxes[j].score < outSortedBoxes[j+1].score) ||
                (outSortedBoxes[j].score == outSortedBoxes[j+1].score && outSortedBoxes[j].batchId > outSortedBoxes[j+1].batchId) ||
                (outSortedBoxes[j].score == outSortedBoxes[j+1].score && outSortedBoxes[j].batchId == outSortedBoxes[j+1].batchId &&
                 outSortedBoxes[j].classId > outSortedBoxes[j+1].classId) ||
                (outSortedBoxes[j].score == outSortedBoxes[j+1].score && outSortedBoxes[j].batchId == outSortedBoxes[j+1].batchId &&
                 outSortedBoxes[j].classId == outSortedBoxes[j+1].classId && outSortedBoxes[j].boxId > outSortedBoxes[j+1].boxId)) {
                FUNC_CALL(swap)(&outSortedBoxes[j], &outSortedBoxes[j+1]);
                swapped = true;
            }
        }

        // IF no two elements were swapped by inner loop, then break
        if (swapped == false)
            break;
    }
}

// boxes shape: {num_batches, num_boxes, 4}
// scores shape: {num_batches, num_classes, num_boxes}
KERNEL (non_max_suppression_ref)(
    const __global INPUT0_TYPE *boxes
    , const __global INPUT1_TYPE *scores
    , __global OUTPUT_TYPE *output
    , __global uchar *buffer0
    , __global uchar *buffer1
    , __global uchar *buffer2
    , __global int *buffer3
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
    #if SELECTED_SCORES_TERM
    , __global SELECTED_SCORES_TYPE *selected_scores
    #endif
    #if VALID_OUTPUTS_TERM
    , __global VALID_OUTPUTS_TYPE *valid_outputs
    #endif

    )
{

#ifdef IS_ZERO_ITER
    int batchId = get_global_id(0);
    int classId = get_global_id(1);

    __global SBOX_INFO *sortedBoxList = (__global SBOX_INFO*)&buffer0[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];
    int sortedBoxNum = FUNC_CALL(initBoxList)(sortedBoxList, NUM_BOXES, scores, SCORE_THRESHOLD_VAL, batchId, classId);
    buffer3[batchId * NUM_CLASSES + classId] = sortedBoxNum;

#elif IS_FIRST_ITER
    int batchId = get_global_id(0);
    int classId = get_global_id(1);

    int sortedBoxNum = buffer3[batchId * NUM_CLASSES + classId];
    __global SBOX_INFO *sortedBoxList = (__global SBOX_INFO*)&buffer0[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];
    FUNC_CALL(quickSortIterative)(sortedBoxList, 0, sortedBoxNum - 1);

#elif IS_SECOND_ITER
    // printf("Is second iter\n");
    int batchId = get_global_id(0);
    int classId = get_global_id(1);

    float scale = 0.0f;
    if (SOFT_NMS_SIGMA_VAL > 0.0f)
    {
        scale = -0.5f / SOFT_NMS_SIGMA_VAL;
    }

    //printf("NUM_SELECT_PER_CLASS_VAL(%d), IOU_THRESHOLD_VAL(%.2f)\n"
    //        "SCORE_THRESHOLD_VAL(%.2f) SOFT_NMS_SIGMA_VAL(%.2f)\n",
    //    NUM_SELECT_PER_CLASS_VAL, IOU_THRESHOLD_VAL,
    //    SCORE_THRESHOLD_VAL, SOFT_NMS_SIGMA_VAL);

    int outputIdx = 0;

    __global SBOX_INFO *sortedBoxList = (__global SBOX_INFO*)&buffer0[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];
    int sortedBoxNum = buffer3[batchId * NUM_CLASSES + classId];

    __global BOX_INFO *selectedBoxList = (__global BOX_INFO*)&buffer1[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];
    int selectedBoxNum = 0;
    while (sortedBoxNum != 0 && selectedBoxNum < NUM_SELECT_PER_CLASS_VAL) {
        //printf("sortedBoxNum(%d) selectedBoxNum(%d) NUM_SELECT_PER_CLASS_VAL(%d)\n",
        //    sortedBoxNum, selectedBoxNum, NUM_SELECT_PER_CLASS_VAL);

        SBOX_INFO next_candidate = sortedBoxList[sortedBoxNum - 1];
        INPUT1_TYPE original_score = next_candidate.score;
        --sortedBoxNum;

        bool should_hard_suppress = false;
        for (int j = selectedBoxNum - 1;
                j >= next_candidate.suppress_begin_index;
                --j)
        {
            const float iou = FUNC_CALL(intersectionOverUnion)(boxes, batchId, next_candidate.boxId, batchId, selectedBoxList[j].boxId);
            next_candidate.score *= FUNC_CALL(scaleIOU)(iou, IOU_THRESHOLD_VAL, scale);;
            // printf("[%d, %d] iou(%f) next.score(%f)  next(%d)-vs-selt(%d)  %f %f\n"
            //     , batchId, classId, iou, next_candidate.score, next_candidate.boxId, selectedBoxList[j].boxId, IOU_THRESHOLD_VAL, scale);

            if (iou >= IOU_THRESHOLD_VAL)
            {
                should_hard_suppress = true;
                break;
            }

            if (next_candidate.score <= SCORE_THRESHOLD_VAL)
            {
                //printf("[%d/%d/%d] iou(%.2f) next.score(%.2f) SCORE_THRESHOLD_VAL(%.2f)\n", batchId, classId, next_candidate.boxId, iou, next_candidate.score, SCORE_THRESHOLD_VAL);
                break;
            }
        }

        next_candidate.suppress_begin_index = selectedBoxNum;

        if (!should_hard_suppress)
        {
            if (next_candidate.score == original_score)
            {
                BOX_INFO binfo;
                binfo.batchId = batchId;
                binfo.classId = classId;
                binfo.boxId = next_candidate.boxId;
                binfo.score = next_candidate.score;
                selectedBoxList[selectedBoxNum] = binfo;
                ++selectedBoxNum;
                continue;
            }
            if (next_candidate.score > SCORE_THRESHOLD_VAL)
            {
                sortedBoxList[sortedBoxNum] = next_candidate;
                ++sortedBoxNum;
                FUNC_CALL(quickSortIterative)(sortedBoxList, 0, sortedBoxNum - 1);
            }
        }
    }

    // Set pad value to indicate the end of selected box list.
    if (selectedBoxNum < NUM_BOXES) {
        selectedBoxList[selectedBoxNum].batchId = -1;
    }

#elif IS_THIRD_ITER
    int outputIdx = 0;
    __global BOX_INFO *sortedBoxList = (__global BOX_INFO*)&buffer2[0];
    for (int batchId = 0; batchId < NUM_BATCHES; batchId++) {
        for (int classId = 0; classId < NUM_CLASSES; classId++) {
            __global BOX_INFO *selectedBoxList = (__global BOX_INFO*)&buffer1[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];
            for (int i = 0; i < NUM_BOXES; i++) {
                if (selectedBoxList[i].batchId > -1) {
                    sortedBoxList[outputIdx] = selectedBoxList[i];
                    outputIdx++;
                } else {
                    break;
                }
            }
        }
    }

#if SORT_RESULT_DESCENDING == 1
    FUNC_CALL(sortOutputBoxList)(sortedBoxList, outputIdx);
#endif
    int output_num = outputIdx;
    unroll_for (int i = 0; i < output_num; i++) {
        const int offset = 3 * i;
        output[offset + 0] = sortedBoxList[i].batchId;
        output[offset + 1] = sortedBoxList[i].classId;
        output[offset + 2] = sortedBoxList[i].boxId;
    }

    // Padding
    unroll_for (int i = output_num; i < OUTPUT_NUM; i++) {
        const int offset = 3 * i;
        output[offset + 0] = -1;
        output[offset + 1] = -1;
        output[offset + 2] = -1;
    }

#endif
}

#undef unroll_for
#undef NUM_BATCHES
#undef NUM_BOXES
#undef NUM_CLASSES
