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
#include "include/fetch_data.cl"

/// optional input variables
/// NUM_SELECT_PER_CLASS_VAL TO_UNIT_TYPE(num_select_per_class[0]),   default is 0 
/// IOU_THRESHOLD_VAL        TO_ACCUMULATOR_TYPE(iou_threshold[0]),   default is ACCUMULATOR_VAL_ZERO
/// SCORE_THRESHOLD_VAL      TO_ACCUMULATOR_TYPE(score_threshold[0]), default is ACCUMULATOR_VAL_ZERO
/// SOFT_NMS_SIGMA_VAL       TO_ACCUMULATOR_TYPE(soft_nms_sigma[0]),  default is ACCUMULATOR_VAL_ZERO
/// OUTPUT_NUM               Number of outputs. [OUTPUT_NUM, 3, 1, 1]
/// BUFFER_STRIDE            20 bytes * NUM_BOXES

#define unroll_for __attribute__((opencl_unroll_hint)) for

#define NUM_BATCHES     INPUT0_BATCH_NUM
#define NUM_BOXES       INPUT0_FEATURE_NUM
#define NUM_CLASSES     INPUT1_FEATURE_NUM

typedef struct {
    ushort boxId;
    int suppress_begin_index;
    INPUT1_TYPE score;
} FUNC(SortedBoxInfo);

typedef struct {
    short batchId;
    ushort classId;
    ushort boxId;
    INPUT1_TYPE score;
} FUNC(BoxInfo);

#define SBOX_INFO FUNC(SortedBoxInfo)
#define BOX_INFO FUNC(BoxInfo)

inline float FUNC(intersectionOverUnion)(const __global INPUT0_TYPE *boxes,
    const short batchA, const ushort boxIdA,
    const short batchB, const ushort boxIdB)
{
    const float4 pA = convert_float4(vload4(0, &boxes[INPUT0_GET_INDEX(batchA, boxIdA, 0, 0)]));
    const float4 pB = convert_float4(vload4(0, &boxes[INPUT0_GET_INDEX(batchB, boxIdB, 0, 0)]));

#if BOX_ENCODING == 0
    const float ax1 = min(pA[1], pA[3]);
    const float ax2 = max(pA[1], pA[3]);
    const float ay1 = min(pA[0], pA[2]);
    const float ay2 = max(pA[0], pA[2]);
    const float bx1 = min(pB[1], pB[3]);
    const float bx2 = max(pB[1], pB[3]);
    const float by1 = min(pB[0], pB[2]);
    const float by2 = max(pB[0], pB[2]);

    /// CORNER
    const float areaA = (ax2 - ax1) * (ay2 - ay1);
    const float areaB = (bx2 - bx1) * (by2 - by1);

    const float intersection_ymin = max(ay1, by1);
    const float intersection_xmin = max(ax1, bx1);
    const float intersection_ymax = min(ay2, by2);
    const float intersection_xmax = min(ax2, bx2);
#else
    /// CENTER
    const float areaA = pA[3] * pA[2];
    const float areaB = pB[3] * pB[2];
    const float halfWidthA = pA[2] / 2;
    const float halfHeightA = pA[3] / 2;
    const float halfWidthB = pB[2] / 2;
    const float halfHeightB = pB[3] / 2;

    const float intersection_ymin = max(pA[1] - halfHeightA, pB[1] - halfHeightB);
    const float intersection_xmin = max(pA[0] - halfWidthA,  pB[0] - halfWidthB);
    const float intersection_ymax = min(pA[1] + halfHeightA, pB[1] + halfHeightB);
    const float intersection_xmax = min(pA[0] + halfWidthA,  pB[0] + halfWidthB);
#endif

    if (areaA <= 0.0f || areaB <= 0.0f)
        return 0.0f;

    const float intersection_area = max(intersection_xmax - intersection_xmin, 0.f) *
                                    max(intersection_ymax - intersection_ymin, 0.f);
    const float union_area = areaA + areaB - intersection_area;
    return intersection_area / union_area;
}

inline float FUNC(scaleIOU)(float iou, float iou_threshold, float scale)
{
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
    const INPUT1_TYPE pivotScore = arr[h].score;
    const ushort pivotBoxId = arr[h].boxId;
    int i = (l - 1);
    for (int j = l; j <= h - 1; j++) {
        if ((arr[j].score < pivotScore) || (arr[j].score == pivotScore && arr[j].boxId > pivotBoxId)) {
            i++;
            FUNC_CALL(swap_sbox_info)(&arr[i], &arr[j]);
        }
    }
    FUNC_CALL(swap_sbox_info)(&arr[i + 1], &arr[h]);
    return (i + 1);
}

inline void FUNC(bubbleSortIterative)(__global SBOX_INFO* arr, int l, int h)
{
    for (int i = 0; i < h-l; i++) {
        bool swapped = false;
        for (int j = l; j < h-i; j++) {
            if ((arr[j].score > arr[j+1].score) || (arr[j].score == arr[j+1].score && arr[j].boxId < arr[j+1].boxId)) {
                FUNC_CALL(swap_sbox_info)(&arr[j], &arr[j+1]);
                swapped = true;
            }
        }

        if (!swapped)
            break;
    }
}

inline void FUNC(quickSortIterative)(__global SBOX_INFO* arr, int l, int h)
{
    // Create an auxiliary stack
    const int kStackSize = 100;
    int stack[kStackSize];

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
        const int p = FUNC_CALL(partition)(arr, l, h);
  
        // If there are elements on left side of pivot,
        // then push left side to stack
        if (p - 1 > l) {
            if (top >= (kStackSize - 1)) {
                FUNC_CALL(bubbleSortIterative)(arr, l, p - 1);
            } else {
                stack[++top] = l;
                stack[++top] = p - 1;
            }
        }

        // If there are elements on right side of pivot,
        // then push right side to stack
        if (p + 1 < h) {
            if (top >= (kStackSize - 1)) {
                FUNC_CALL(bubbleSortIterative)(arr, p + 1, h);
            } else {
                stack[++top] = p + 1;
                stack[++top] = h;
            }
        }
    }
}

inline int FUNC(initBoxList)(__global SBOX_INFO *outBoxes, int boxNum, const __global INPUT1_TYPE *scores, float score_threshold, short batchId, ushort classId)
{
    int count = 0;
    for (ushort i = 0; i < boxNum; ++i) {
        const INPUT1_TYPE score = scores[INPUT1_GET_INDEX(batchId, classId, i, 0)];
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
        const int outputId = i * 3;
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

// KERNEL_ARGs
//
// boxes
//  - shape: {num_batches, num_boxes, 4}
// scores
//  - shape: {num_batches, num_classes, num_boxes}
// buffer0 (intermediate buffer)
//  - size: batch_num * class_num * boxes_num * sizeof(SBOX_INFO)
//  - desc: filtered and sorted SBOX_INFO list
// buffer1 (intermediate buffer)
//  - size: batch_num * class_num * boxes_num * sizeof(BOX_INFO)
//  - desc: selected SBOX_INFO list by iou calucation
// buffer2 (intermediate buffer)
//  - size: batch_num * class_num * boxes_num * sizeof(BOX_INFO)
//  - desc: selected SBOX_INFO list by iou calucation
// buffer3 (intermediate buffer)
//  - size: batch_num * class_num * 4
//  - desc: sorted box num for batch*class
#ifdef IS_STAGE_0
KERNEL (non_max_suppression_ref_stage_0)(
    const __global INPUT1_TYPE *scores
    , __global uchar *buffer0
    , __global int *buffer3
    #ifdef SCORE_THRESHOLD_TYPE
    , const __global SCORE_THRESHOLD_TYPE *score_threshold
    #endif
    )
{
    const short batchId = get_global_id(0);
    const ushort classId = get_global_id(1);
    const ushort box_gid = get_global_id(2);

    const int start_bid = box_gid * NUM_SCORE_PER_ITEM;
    const int end_bid = min(start_bid + NUM_SCORE_PER_ITEM, NUM_BOXES);

    __local char bit_mask[NUM_BIT_MASK];
    __local int block_num[NUM_SCORE_BLOCK];

    block_num[box_gid] = 0;

    {
        int mask_id = start_bid / 8;
        int total_block_selected_num = 0;
        for (int i = start_bid; i < end_bid; i += 8) {
            MAKE_VECTOR_TYPE(INPUT1_TYPE, 8) score8 = vload8(0, &scores[INPUT1_GET_INDEX(batchId, classId, i, 0)]);

            char mask = 0;
            for (int bi = 0; bi < 8; bi++) {
                if ((i + bi) >= NUM_BOXES)
                    break;

                if (convert_float(score8[bi]) <= SCORE_THRESHOLD_VAL)
                    continue;

                mask |= (1 << bi);
                total_block_selected_num++;
            }
            bit_mask[mask_id] = mask;
            mask_id++;
        }

        block_num[box_gid] = total_block_selected_num;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    {
        // first item of group
        if (box_gid == 0 && get_local_id(2) == 0) {
            int acc_num = 0;
            int total_sel_num = 0;
            for (int i = 0; i < NUM_SCORE_BLOCK; i++) {
                int n = block_num[i];
                block_num[i] = acc_num;
                acc_num += n;
            }
            buffer3[batchId * NUM_CLASSES + classId] = acc_num;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    {
        __global SBOX_INFO *sortedBoxList = (__global SBOX_INFO*)&buffer0[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];

        int write_offset = block_num[box_gid];

        int mask_id = start_bid / 8;
        for (int i = start_bid; i < end_bid; i += 8) {
            MAKE_VECTOR_TYPE(INPUT1_TYPE, 8) score8 = vload8(0, &scores[INPUT1_GET_INDEX(batchId, classId, i, 0)]);
            const char mask = bit_mask[mask_id];

            for (int bi = 0; bi < 8; bi++) {
                if ((mask & (1 << bi)) && (i + bi) < NUM_BOXES) {
                    SBOX_INFO binfo;
                    binfo.boxId = i + bi;
                    binfo.suppress_begin_index = 0;
                    binfo.score = score8[bi];
                    sortedBoxList[write_offset] = binfo;

                    write_offset++;
                }
            }
            mask_id++;
        }
    }
}
#endif /* IS_STAGE_0 */

#ifdef IS_STAGE_1

#if LOCAL_BATCH_NUM != 1
#error "The batch number of LWS should be 1."
#endif

KERNEL (non_max_suppression_ref_stage_1)(
    __global uchar *buffer0
    , __global int *buffer3
    )
{
    const int batchId = get_global_id(0);
    const int classId = get_global_id(1);
    const int workItemId = get_global_id(2);
    const int localClassId = get_local_id(1);
    __local int __range[LOCAL_CLASS_NUM][LOCAL_WORK_NUM * 2];

    const int sortedBoxNum = buffer3[batchId * NUM_CLASSES + classId];
    __global SBOX_INFO *sortedBoxList = (__global SBOX_INFO*)&buffer0[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];
    if (workItemId == 0) {
        __range[localClassId][0] = 0;
        __range[localClassId][1] = sortedBoxNum - 1;
    } else {
        __range[localClassId][workItemId * 2] = 0;
        __range[localClassId][workItemId * 2 + 1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int range_step = 2;
    const int first_id = workItemId * 2;
    for (int i = 0; i < PARTITION_STEP; ++i, range_step *= 2) {
        if (workItemId <= i) {
            const int begin_id = __range[localClassId][first_id];
            const int end_id = __range[localClassId][first_id + 1];
            const int second_id = first_id + range_step;

            if (begin_id < end_id) {
                const int pivot = FUNC_CALL(partition)(sortedBoxList, begin_id, end_id);
                __range[localClassId][first_id     ] = begin_id;
                __range[localClassId][first_id + 1 ] = max(pivot - 1, begin_id);
                __range[localClassId][second_id    ] = min(pivot + 1, end_id);
                __range[localClassId][second_id + 1] = end_id;
            }
        }

        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }

    const int begin_id = __range[localClassId][first_id];
    const int end_id = __range[localClassId][first_id + 1];
    if (begin_id < end_id) {
        FUNC_CALL(quickSortIterative)(sortedBoxList, begin_id, end_id);
    }
}
#endif /* IS_STAGE_1 */

#ifdef IS_STAGE_2
KERNEL (non_max_suppression_ref_stage_2)(
    const __global INPUT0_TYPE *boxes
    , __global uchar *buffer0
    , __global uchar *buffer1
    , __global int *buffer3
    #ifdef NUM_SELECT_PER_CLASS_TYPE
    , const __global NUM_SELECT_PER_CLASS_TYPE *num_select_per_class
    #endif
    #ifdef IOU_THRESHOLD_TYPE
    , const __global IOU_THRESHOLD_TYPE *iou_threshold
    #endif
    #ifdef SCORE_THRESHOLD_TYPE
    , const __global SCORE_THRESHOLD_TYPE *score_threshold
    #endif
    #ifdef SOFT_NMS_SIGMA_TYPE
    , const __global SOFT_NMS_SIGMA_TYPE *soft_nms_sigma
    #endif
    )
{
    const short batchId = get_global_id(0);
    const ushort classId = get_global_id(1);

    float scale = 0.0f;
    if (SOFT_NMS_SIGMA_VAL > 0.0f) {
        scale = -0.5f / SOFT_NMS_SIGMA_VAL;
    }

    __global SBOX_INFO *sortedBoxList = (__global SBOX_INFO*)&buffer0[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];
    int sortedBoxNum = buffer3[batchId * NUM_CLASSES + classId];

    __global BOX_INFO *selectedBoxList = (__global BOX_INFO*)&buffer1[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];
    int selectedBoxNum = 0;
    while (sortedBoxNum != 0 && selectedBoxNum < NUM_SELECT_PER_CLASS_VAL) {
        SBOX_INFO next_candidate = sortedBoxList[sortedBoxNum - 1];
        INPUT1_TYPE original_score = next_candidate.score;
        --sortedBoxNum;

        bool should_hard_suppress = false;
        for (int j = selectedBoxNum - 1;
                j >= next_candidate.suppress_begin_index;
                --j) {
            const float iou = FUNC_CALL(intersectionOverUnion)(boxes, batchId, next_candidate.boxId, batchId, selectedBoxList[j].boxId);
            next_candidate.score *= FUNC_CALL(scaleIOU)(iou, IOU_THRESHOLD_VAL, scale);

            if (iou >= IOU_THRESHOLD_VAL) {
                should_hard_suppress = true;
                break;
            }

            if (convert_float(next_candidate.score) <= SCORE_THRESHOLD_VAL) {
                break;
            }
        }

        next_candidate.suppress_begin_index = selectedBoxNum;

        if (!should_hard_suppress) {
            if (next_candidate.score == original_score) {
                BOX_INFO binfo;
                binfo.batchId = batchId;
                binfo.classId = classId;
                binfo.boxId = next_candidate.boxId;
                binfo.score = next_candidate.score;
                selectedBoxList[selectedBoxNum] = binfo;
                ++selectedBoxNum;

                continue;
            }

            if (convert_float(next_candidate.score) > SCORE_THRESHOLD_VAL) {
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
}
#endif /* IS_STAGE_2 */

#ifdef IS_STAGE_FINAL
KERNEL (non_max_suppression_ref_stage_final)(
    __global OUTPUT_TYPE *output
    , __global uchar *buffer1
    , __global uchar *buffer2
    #ifdef SECOND_OUTPUT_TYPE
    , __global SECOND_OUTPUT_TYPE *selected_scores
    #endif
    #ifdef THIRD_OUTPUT_TYPE
    , __global THIRD_OUTPUT_TYPE *valid_outputs
    #endif
    )
{
    int outputIdx = 0;
    __global BOX_INFO *sortedBoxList = (__global BOX_INFO*)&buffer2[0];
    for (short batchId = 0; batchId < NUM_BATCHES; batchId++) {
        for (ushort classId = 0; classId < NUM_CLASSES; classId++) {
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

    unroll_for (int i = 0; i < outputIdx; i++) {
        const int offset = 3 * i;
        output[offset + 0] = sortedBoxList[i].batchId;
        output[offset + 1] = sortedBoxList[i].classId;
        output[offset + 2] = sortedBoxList[i].boxId;
    }

    // Padding
    unroll_for (int i = outputIdx; i < OUTPUT_NUM; i++) {
        const int offset = 3 * i;
        output[offset + 0] = -1;
        output[offset + 1] = -1;
        output[offset + 2] = -1;
    }

#ifdef SECOND_OUTPUT_TYPE
    unroll_for (int i = 0; i < outputIdx; i++) {
        const int offset = 3 * i;
        selected_scores[offset + 0] = TO_SECOND_OUTPUT_TYPE(sortedBoxList[i].batchId);
        selected_scores[offset + 1] = TO_SECOND_OUTPUT_TYPE(sortedBoxList[i].classId);
        selected_scores[offset + 2] = TO_SECOND_OUTPUT_TYPE(sortedBoxList[i].score);
    }

    // Padding
    unroll_for (int i = outputIdx; i < OUTPUT_NUM; i++) {
        const int offset = 3 * i;
        selected_scores[offset + 0] = -1;
        selected_scores[offset + 1] = -1;
        selected_scores[offset + 2] = -1;
    }
#endif

#ifdef THIRD_OUTPUT_TYPE
    valid_outputs[0] = TO_THIRD_OUTPUT_TYPE(outputIdx);
#endif
}
#endif  /* IS_STAGE_FINAL */
