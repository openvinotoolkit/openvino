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

#define PRINT printf
//#define PRINT

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
    const int batchId = get_global_id(0);
    const int classId = get_global_id(1);
    const int box_gid = get_global_id(2);

    const int start_bid = box_gid * NUM_SCORE_PER_ITEM;
    const int end_bid = min(start_bid + NUM_SCORE_PER_ITEM, NUM_BOXES);

    //printf("IS_ZERO_ITER-1 [%3d][%3d][%3d]\n", batchId, classId, box_gid);

    //printf("batchId[%d] classId[%d] box_gid[%d] start_bid[%d] end_bid[%d] %d\n", batchId, classId, box_gid, start_bid, end_bid, NUM_SCORE_PER_ITEM);

    __local char bit_mask[NUM_BIT_MASK];
    __local int block_num[NUM_SCORE_BLOCK];
    // printf("(%3d %3d %3d) (%3zu %3zu %3zu) (%3zu %3zu %3zu) \n", batchId, classId, box_gid
    // , get_group_id(0), get_group_id(1), get_group_id(2)
    // , get_local_id(0), get_local_id(1), get_local_id(2));

    block_num[box_gid] = 0;

    {
        int mask_id = start_bid / 8;
        int total_block_selected_num = 0;
        for (int i = start_bid; i < end_bid; i += 8) {
            MAKE_VECTOR_TYPE(INPUT1_TYPE, 8) score8 = vload8(0, &scores[INPUT1_GET_INDEX(batchId, classId, i, 0)]);

            char mask = 0;
            for (int bi = 0; bi < 8; bi++) {
                if (TO_ACCUMULATOR_TYPE(score8[bi]) < SCORE_THRESHOLD_VAL || (i + bi) >= NUM_BOXES)
                    continue;
                // printf("[%d %d %d] %f (%f) %d\n", batchId, classId, i + bi, TO_ACCUMULATOR_TYPE(score8[bi]), SCORE_THRESHOLD_VAL, NUM_BOXES);
                mask |= (1 << bi);
                total_block_selected_num++;
            }
            bit_mask[mask_id] = mask;
            //printf(" [%3d][%3d][] i(%d) mask_id(%d) mask(%d)\n", batchId, classId, i, mask_id, mask);
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
            // printf("[%3d %3d] %d\n", batchId, classId, acc_num);
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
                if ((mask & (1 << bi)) && (i + bi) < NUM_BOXES)
                {
                    SBOX_INFO binfo;
                    binfo.boxId = i + bi;
                    binfo.suppress_begin_index = 0;
                    binfo.score = score8[bi];
                    sortedBoxList[write_offset] = binfo;
                    //printf("{K} first1 [%3d][%3d][%3d] score(%.2f)\n", batchId, classId, binfo.boxId, binfo.score);
                    write_offset++;
                }
                else
                {
                    //printf("{K} [%3d][%3d][???] first2 mask_id(%d) mask(%d) i(%d) bi(%d) NUM_BOXES(%d)\n", batchId, classId, mask_id, mask, i, bi, NUM_BOXES);
                }
            }
            mask_id++;
        }
    }
    //printf("IS_ZERO_ITER-END [%3d][%3d][%3d]\n", batchId, classId, box_gid);
#elif IS_FIRST_ITER
    int batchId = get_global_id(0);
    int classId = get_global_id(1);
    int workItemId = get_global_id(2);
    int localClassId = get_local_id(1);
    __local int __range[LOCAL_CLASS_NUM][LOCAL_WORK_NUM * 4];

    PRINT("IS_FIRST_ITER-1 [%3d][%3d][%3d]\n", batchId, classId, workItemId);

    int sortedBoxNum = buffer3[batchId * NUM_CLASSES + classId];
    __global SBOX_INFO *sortedBoxList = (__global SBOX_INFO*)&buffer0[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];
    if (workItemId == 0 && sortedBoxNum > 1) {
        __range[localClassId][0] = 0;
        __range[localClassId][1] = sortedBoxNum - 1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int range_step = 2;
    int first_id = workItemId * 2;
    for (int i = 0; i < PARTITION_STEP; ++i, range_step *= 2) {
        if (sortedBoxNum > 1 && workItemId <= i) {
            int begin_id = __range[localClassId][first_id];
            int end_id = __range[localClassId][first_id + 1];
            int second_id = first_id + range_step;

            if (begin_id < end_id) {
                //printf("[%d/%d/%d] partition[%d ~ %d] sortedBoxNum(%d) i(%d)\n",
                //    batchId, classId, workItemId, begin_id, end_id, sortedBoxNum, i);

                int pivot = FUNC_CALL(partition)(sortedBoxList, begin_id, end_id);
                //printf("[%d/%d/%d] partition[%d ~ %d] sortedBoxNum(%d) i(%d) pivot(%d) done\n",
                //    batchId, classId, workItemId, begin_id, end_id, sortedBoxNum, i, pivot);
                __range[localClassId][first_id     ] = begin_id;
                __range[localClassId][first_id + 1 ] = max(pivot - 1, begin_id);
                __range[localClassId][second_id    ] = min(pivot + 1, end_id);
                __range[localClassId][second_id + 1] = end_id;
            } else {
                __range[localClassId][second_id    ] = 0;
                __range[localClassId][second_id + 1] = 0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //printf("IS_FIRST_ITER-2 [%3d][%3d][%3d]\n", batchId, classId, workItemId);
    //barrier(CLK_LOCAL_MEM_FENCE);

    if (sortedBoxNum > 1) {
        int begin_id = __range[localClassId][first_id];
        int end_id = __range[localClassId][first_id + 1];
        if (begin_id < end_id) {
            //printf("[%d/%d/%d] quickSortIterative[%d ~ %d] sortedBoxNum(%d)\n",
            //    batchId, classId, workItemId, begin_id, end_id, sortedBoxNum);
            FUNC_CALL(quickSortIterative)(sortedBoxList, begin_id, end_id);
            //printf("[%d/%d/%d] quickSortIterative[%d ~ %d] sortedBoxNum(%d) done.\n",
            //    batchId, classId, workItemId, begin_id, end_id, sortedBoxNum);
        }
    }
    PRINT("IS_FIRST_ITER-END [%3d][%3d][%3d]\n", batchId, classId, workItemId);
#elif IS_SECOND_ITER
    int batchId = get_global_id(0);
    int classId = get_global_id(1);
    PRINT("IS_SECOND_ITER-1 [%3d][%3d][0]\n", batchId, classId);

    float scale = 0.0f;
    if (SOFT_NMS_SIGMA_VAL > 0.0f)
    {
        scale = -0.5f / SOFT_NMS_SIGMA_VAL;
    }

    //if (batchId == 0 && classId == 0)
    //    printf("{K} NUM_SELECT_PER_CLASS_VAL(%d), IOU_THRESHOLD_VAL(%.2f)\n"
    //            "{K} SCORE_THRESHOLD_VAL(%.2f) SOFT_NMS_SIGMA_VAL(%.2f)\n",
    //        NUM_SELECT_PER_CLASS_VAL, IOU_THRESHOLD_VAL,
    //        SCORE_THRESHOLD_VAL, SOFT_NMS_SIGMA_VAL);

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
            next_candidate.score *= FUNC_CALL(scaleIOU)(iou, IOU_THRESHOLD_VAL, scale);
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
    //printf("{K} selectedBoxNum(%d) NUM_BOXES(%ld)\n", selectedBoxNum, NUM_BOXES);
    if (selectedBoxNum < NUM_BOXES) {
        selectedBoxList[selectedBoxNum].batchId = -1;
    }
    PRINT("IS_SECOND_ITER-END [%3d][%3d][0]\n", batchId, classId);
#elif IS_THIRD_ITER
    PRINT("IS_THIRD_ITER-1\n");
    int outputIdx = 0;
    __global BOX_INFO *sortedBoxList = (__global BOX_INFO*)&buffer2[0];
    for (int batchId = 0; batchId < NUM_BATCHES; batchId++) {
        for (int classId = 0; classId < NUM_CLASSES; classId++) {
            __global BOX_INFO *selectedBoxList = (__global BOX_INFO*)&buffer1[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];
            for (int i = 0; i < NUM_BOXES; i++) {
                if (selectedBoxList[i].batchId > -1) {
                    //printf("{K} third_copy %3d/%3d/%3d - score(%.2f)\n", selectedBoxList[i].batchId, selectedBoxList[i].classId, selectedBoxList[i].boxId, selectedBoxList[i].score);
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
    //printf("{K}   output_num: %d\n", output_num);
    unroll_for (int i = 0; i < output_num; i++) {
        const int offset = 3 * i;
        output[offset + 0] = sortedBoxList[i].batchId;
        output[offset + 1] = sortedBoxList[i].classId;
        output[offset + 2] = sortedBoxList[i].boxId;
        //printf("{K}   [%3ld] %3d/%3d/%3d\n", offset, output[offset + 0], output[offset + 1], output[offset + 2]);
    }

    // Padding
    unroll_for (int i = output_num; i < OUTPUT_NUM; i++) {
        const int offset = 3 * i;
        output[offset + 0] = -1;
        output[offset + 1] = -1;
        output[offset + 2] = -1;
    }

    PRINT("IS_THIRD_ITER-END\n");
#endif
}

#undef unroll_for
#undef NUM_BATCHES
#undef NUM_BOXES
#undef NUM_CLASSES
