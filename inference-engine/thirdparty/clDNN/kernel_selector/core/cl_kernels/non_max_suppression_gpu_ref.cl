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

typedef struct {
    int batchId;
    int classId;
    int boxId;
    int suppress_begin_index;
    INPUT1_TYPE score;
} FUNC(BoxInfo);

#define BOX_INFO FUNC(BoxInfo)

inline float FUNC(intersectionOverUnion)(const __global INPUT0_TYPE *boxes,
    int batchA, int boxIdA,
    int batchB, int boxIdB)
{
    float4 pA = convert_float4(vload4(0, &boxes[INPUT0_GET_INDEX(batchA, boxIdA, 0, 0)]));
    float4 pB = convert_float4(vload4(0, &boxes[INPUT0_GET_INDEX(batchB, boxIdB, 0, 0)]));

    float areaA = (pA[3] - pA[1]) * (pA[2] - pA[0]);
    float areaB = (pB[3] - pB[1]) * (pB[2] - pB[0]);

    float intersection_ymin = max(pA[1], pB[1]);
    float intersection_xmin = max(pA[0], pB[0]);
    float intersection_ymax = min(pA[3], pB[3]);
    float intersection_xmax = min(pA[2], pB[2]);

    float intersection_area =
        max(intersection_ymax - intersection_ymin, 0.0f) *
        max(intersection_xmax - intersection_xmin, 0.0f);

    return intersection_area / (areaA + areaB - intersection_area);
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

inline void FUNC(swap)(__global BOX_INFO* a, __global BOX_INFO* b)
{
    BOX_INFO temp = *a;
    *a = *b;
    *b = temp;
}

inline int FUNC(initBoxList)(__global BOX_INFO *outBoxes, int boxNum, const __global INPUT1_TYPE *scores, float score_threshold, int batchId, int classId)
{
    int count = 0;
    for (int i = 0; i < boxNum; ++i) {
        float score = scores[INPUT1_GET_INDEX(batchId, classId, i, 0)];
        if (score < score_threshold) continue;

        // printf("score: %f, threshold: %f\n", score, score_threshold);

        outBoxes[count].batchId = batchId;
        outBoxes[count].classId = classId;
        outBoxes[count].boxId = i;
        outBoxes[count].suppress_begin_index = 0;
        outBoxes[count].score = scores[INPUT1_GET_INDEX(batchId, classId, i, 0)];
        ++count;
    }

    return count;
}

inline void FUNC(sortBoxList)(__global BOX_INFO *outSortedBoxes, int boxNum)
{
    for (int i = 0; i < boxNum - 1; ++i) {
        // bool swapped = false;
        for (int j = 0; j < boxNum - i - 1; ++j) {
            if (outSortedBoxes[j].score > outSortedBoxes[j+1].score) {
                FUNC_CALL(swap)(&outSortedBoxes[j], &outSortedBoxes[j+1]);
                // swapped = true;
            }
        }

        // // IF no two elements were swapped by inner loop, then break
        // if (swapped == false)
        //     break;
    }
}

inline void FUNC(initOutputBoxList)(__global BOX_INFO *outBoxes, int boxNum, const __global INPUT1_TYPE *scores, __global OUTPUT_TYPE *output)
{
    for (int i = 0; i < boxNum; ++i) {
        int outputId = i * 3;
        outBoxes[i].batchId = output[outputId + 0];
        outBoxes[i].classId = output[outputId + 1];
        outBoxes[i].boxId = output[outputId + 2];
        outBoxes[i].suppress_begin_index = 0;
        outBoxes[i].score = scores[INPUT1_GET_INDEX(outBoxes[i].batchId, outBoxes[i].classId, outBoxes[i].boxId, 0)];
    }
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

#define NUM_BATCHES     INPUT0_BATCH_NUM
#define NUM_BOXES       INPUT0_FEATURE_NUM
#define NUM_CLASSES     INPUT1_FEATURE_NUM

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

#ifdef IS_FIRST_ITER
    int batchId = get_global_id(0);
    int classId = get_global_id(1);

    __global BOX_INFO *sortedBoxList = (__global BOX_INFO*)&buffer0[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];
    int sortedBoxNum = FUNC_CALL(initBoxList)(sortedBoxList, NUM_BOXES, scores, SCORE_THRESHOLD_VAL, batchId, classId);
    // printf("Sorted Box Num: %d\n", sortedBoxNum);
    FUNC_CALL(sortBoxList)(sortedBoxList, sortedBoxNum);

    buffer3[batchId * NUM_CLASSES + classId] = sortedBoxNum;

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

    __global BOX_INFO *sortedBoxList = (__global BOX_INFO*)&buffer0[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];
    int sortedBoxNum = buffer3[batchId * NUM_CLASSES + classId];

    __global BOX_INFO *selectedBoxList = (__global BOX_INFO*)&buffer1[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];
    int selectedBoxNum = 0;
    INPUT1_TYPE original_score = 0;
    while (sortedBoxNum != 0 && selectedBoxNum < NUM_SELECT_PER_CLASS_VAL) {
        //printf("sortedBoxNum(%d) selectedBoxNum(%d) NUM_SELECT_PER_CLASS_VAL(%d)\n",
        //    sortedBoxNum, selectedBoxNum, NUM_SELECT_PER_CLASS_VAL);

        BOX_INFO next_candidate = sortedBoxList[sortedBoxNum - 1];
        INPUT1_TYPE original_score = next_candidate.score;
        --sortedBoxNum;

        //printf("next_candidate.boxId(%d) next_candidate.score(%.2f)\n", next_candidate.boxId, next_candidate.score);
        //printf("next_candidate.suppress_begin_index(%d)\n", next_candidate.suppress_begin_index);

        bool should_hard_suppress = false;
        for (int j = selectedBoxNum - 1;
                j >= next_candidate.suppress_begin_index;
                --j)
        {
            float iou = FUNC_CALL(intersectionOverUnion)(boxes, batchId, next_candidate.boxId, batchId, selectedBoxList[j].boxId);
            next_candidate.score *= FUNC_CALL(scaleIOU)(iou, IOU_THRESHOLD_VAL, scale);
            //printf("iou(%.2f) next.score(%.2f)  next(%d)-vs-selt(%d)\n", iou, next_candidate.score, next_candidate.boxId, selectedBoxList[j].boxId);

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
                //printf("SEL[ ] batch(%d) classId(%d) boxId(%d) score(%.2f)\n", 
                //    next_candidate.batchId, next_candidate.classId, next_candidate.boxId, next_candidate.score);
                selectedBoxList[selectedBoxNum] = next_candidate;
                ++selectedBoxNum;
                continue;
            }
            if (next_candidate.score > SCORE_THRESHOLD_VAL)
            {
                //printf("called sortBoxList\n");
                sortedBoxList[sortedBoxNum] = next_candidate;
                ++sortedBoxNum;
                FUNC_CALL(sortBoxList)(sortedBoxList, sortedBoxNum);
            }
        }
    }

    // printf("Selected Box Num: %d\n", selectedBoxNum);

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
                    sortedBoxList[outputIdx].batchId = selectedBoxList[i].batchId;
                    sortedBoxList[outputIdx].classId = selectedBoxList[i].classId;
                    sortedBoxList[outputIdx].boxId   = selectedBoxList[i].boxId;
                    sortedBoxList[outputIdx].score   = selectedBoxList[i].score;
                    sortedBoxList[outputIdx].suppress_begin_index = selectedBoxList[i].suppress_begin_index;
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
