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

#include "include/include_all.cl"
#include "include/detection_output_common.cl"


typedef struct {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
} FUNC(BBoxes);

typedef struct {
    int batchId;
    int classId;
    int boxId;
    UNIT_TYPE score;
} FUNC(Scores);

#define BBOXES_INFO FUNC(BBoxes)
#define SCORES_INFO FUNC(Scores)

inline void FUNC(swap_scores_info)(__global SCORES_INFO* a, __global SCORES_INFO* b)
{
    SCORES_INFO temp = *a;
    *a = *b;
    *b = temp;
}

inline int FUNC(partition)(__global SCORES_INFO* arr, int l, int h, bool use_custom_comp)
{
    UNIT_TYPE pivotScore = arr[h].score;
    int pivotBoxId = arr[h].boxId;
    int i = (l - 1);
    for (int j = l; j <= h - 1; j++) {
        if (use_custom_comp) {
            if ((arr[j].score > pivotScore) || (arr[j].score == pivotScore && arr[j].boxId < pivotBoxId)) {
                i++;
                FUNC_CALL(swap_scores_info)(&arr[i], &arr[j]);
            }
        } else {
            if (arr[j].score > pivotScore) {
                i++;
                FUNC_CALL(swap_scores_info)(&arr[i], &arr[j]);
            }
        }
    }
    FUNC_CALL(swap_scores_info)(&arr[i + 1], &arr[h]);
    return (i + 1);
}

inline void FUNC(quickSortIterative)(__global SCORES_INFO* arr, int l, int h, bool use_custom_comp)
{
    // Create an auxiliary stack
    int stack[NUM_OF_PRIORS];

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
        int p = FUNC_CALL(partition)(arr, l, h, use_custom_comp);
  
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

inline float FUNC(jaccardOverlap)(__global BBOXES_INFO* bbox1, __global BBOXES_INFO* bbox2)
{
    //printf("jaccardOverlap | bbox1[%f, %f, %f, %f] bbox2[%f, %f, %f, %f]\n",
    //        bbox1[0].xmin, bbox1[0].ymin, bbox1[0].xmax, bbox1[0].ymax,
    //        bbox2[0].xmin, bbox2[0].ymin, bbox2[0].xmax, bbox2[0].ymax);
    BBOXES_INFO intersectBbox;
    if (bbox2[0].xmin > bbox1[0].xmax || bbox2[0].xmax < bbox1[0].xmin ||
        bbox2[0].ymin > bbox1[0].ymax || bbox2[0].ymax < bbox1[0].ymin) {
        intersectBbox.xmin = 0;
        intersectBbox.ymin = 0;
        intersectBbox.xmax = 0;
        intersectBbox.ymax = 0;
    } else {
        intersectBbox.xmin = max(bbox1[0].xmin, bbox2[0].xmin);
        intersectBbox.ymin = max(bbox1[0].ymin, bbox2[0].ymin);
        intersectBbox.xmax = min(bbox1[0].xmax, bbox2[0].xmax);
        intersectBbox.ymax = min(bbox1[0].ymax, bbox2[0].ymax);
    }

    const float intersectWidth = intersectBbox.xmax - intersectBbox.xmin;
    const float intersectHeight = intersectBbox.ymax - intersectBbox.ymin;
    if (intersectWidth > 0 && intersectHeight > 0) {
        const float intersect_size = intersectWidth * intersectHeight;
        const float bbox1_size = (bbox1[0].xmax - bbox1[0].xmin) * (bbox1[0].ymax - bbox1[0].ymin);
        const float bbox2_size = (bbox2[0].xmax - bbox2[0].xmin) * (bbox2[0].ymax - bbox2[0].ymin);
        return intersect_size / (bbox1_size + bbox2_size - intersect_size);
    } else {
        return 0.f;
    }
}

UNIT_TYPE FUNC(get_confidence_offset)(const uint idx_prior, const uint idx_class, const uint idx_image)
{
    return (idx_prior * NUM_CLASSES + idx_image * NUM_OF_PRIORS * NUM_CLASSES + idx_class) * CONF_XY_SIZE_PRODUCT + CONF_PADDING;
}

UNIT_TYPE FUNC(get_largest_score)(__global UNIT_TYPE* input_confidence, const uint idx_prior, const uint idx_image)
{
    const uint idx_start = (BACKGROUND_LABEL_ID == 0 ? 1 : 0);
    uint offset = FUNC_CALL(get_confidence_offset)(idx_prior, idx_start, idx_image);
    UNIT_TYPE max_score = input_confidence[offset];
    int idx = idx_start;

    for (uint j = idx_start; j < NUM_CLASSES; j++)
    {
        offset = FUNC_CALL(get_confidence_offset)(idx_prior, j, idx_image);
        UNIT_TYPE score = input_confidence[offset];
        if (score > max_score) {
            max_score = score;
            idx = j;
        }
    }
    return idx;
}

#ifdef IS_ZERO_ITER_CAFFE
KERNEL (detection_output_stage_0_caffe)(
    __global UNIT_TYPE* input_location,
    __global UNIT_TYPE* input_confidence,
    __global UNIT_TYPE* input_prior_box,
    __global uchar *buffer0,
    __global uchar *buffer1,
    __global int *buffer3)
{
    //printf("detection_output_stage_0_caffe | NUM_OF_IMAGES=[%3d] NUM_LOC_CLASSES=[%3d]/NUM_CLASSES=[%3d] NUM_OF_PRIORS=[%3d], CONFIDENCE_THRESHOLD=[%f]\n", NUM_OF_IMAGES, NUM_LOC_CLASSES, NUM_CLASSES, NUM_OF_PRIORS, CONFIDENCE_THRESHOLD);

    __global BBOXES_INFO *bboxesList = (__global BBOXES_INFO*)&buffer0[0];
    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer1[0];

    for (uint idx_image = 0; idx_image < NUM_OF_IMAGES; idx_image++)
    {
        for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++)
        {
            int scores_size_offset = idx_image * NUM_CLASSES + idx_class;
            buffer3[scores_size_offset] = 0;
        }
        for (uint idx_prior = 0; idx_prior < NUM_OF_PRIORS; idx_prior++)
        {
            for (uint idx_class = 0; idx_class < NUM_LOC_CLASSES; idx_class++)
            {
                int loc_label = ((SHARE_LOCATION)? 0 : idx_class);
                if (!SHARE_LOCATION && loc_label == BACKGROUND_LABEL_ID)
                {
                    continue;
                }
                int bboxes_offset = (idx_image * NUM_LOC_CLASSES * NUM_OF_PRIORS) + idx_class * NUM_OF_PRIORS + idx_prior;
                UNIT_TYPE decoded_bbox[4];
                FUNC_CALL(get_decoded_bbox)(decoded_bbox, input_location, input_prior_box, idx_prior, idx_class, idx_image);
                BBOXES_INFO bbox_info;
                bbox_info.xmin = decoded_bbox[0];
                bbox_info.ymin = decoded_bbox[1];
                bbox_info.xmax = decoded_bbox[2];
                bbox_info.ymax = decoded_bbox[3];
                bboxesList[bboxes_offset] = bbox_info;
                //printf("bboxesList[%d] = [%f, %f, %f, %f]\n", bboxes_offset, bbox_info.xmin, bbox_info.ymin, bbox_info.xmax, bbox_info.ymax);
            }
            for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++)
            {
                UNIT_TYPE score = FUNC_CALL(get_score)(input_confidence, idx_prior, idx_class, idx_image);
                if (score > 0) {
                    int scores_size_offset = idx_image * NUM_CLASSES + idx_class;
                    int acc_num = buffer3[scores_size_offset];
                    int scores_offset = (idx_image * NUM_CLASSES * NUM_OF_PRIORS) + idx_class * NUM_OF_PRIORS + acc_num;
                    SCORES_INFO score_info;
                    score_info.batchId = idx_image;
                    score_info.classId = idx_class;
                    score_info.boxId = idx_prior;
                    score_info.score = score;
                    scoresList[scores_offset] = score_info;
                    buffer3[scores_size_offset] = (acc_num + 1);
                    //printf("scoresList[%d] = [batchId:%d, classId:%d, boxId:%d, score:%f]\n", scores_offset, score_info.batchId, score_info.classId, score_info.boxId, score_info.score);
                }
            }
        }
        //for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++)
        //{
        //    int scores_size_offset = idx_image * NUM_CLASSES + idx_class;
        //    int acc_num = buffer3[scores_size_offset];
        //    int scores_offset = (idx_image * NUM_CLASSES * NUM_OF_PRIORS) + idx_class * NUM_OF_PRIORS;
        //    printf("detection_output_stage_0_caffe | acc_num[%d]\n", acc_num);
        //    for (uint idx_prior = 0; idx_prior < acc_num; idx_prior++)
        //    {
        //        SCORES_INFO score_info;
        //        score_info = scoresList[scores_offset + idx_prior];
        //        printf("detection_output_stage_0_caffe | scoresList[%d] = [batchId:%d, classId:%d, boxId:%d, score:%f]\n", scores_offset + idx_prior, score_info.batchId, score_info.classId, score_info.boxId, score_info.score);
        //    }
        //}
    }
}
#endif /* IS_ZERO_ITER_CAFFE */

#ifdef IS_ZERO_ITER_CAFFE_OPT
KERNEL (detection_output_stage_0_caffe_opt)(
    __global UNIT_TYPE* input_location,
    __global UNIT_TYPE* input_confidence,
    __global UNIT_TYPE* input_prior_box,
    __global uchar *buffer0,
    __global uchar *buffer1,
    __global int *buffer3)
{
    const int batchId = get_global_id(0);
    const int classId = get_global_id(1);
    //printf("detection_output_stage_0_caffe_opt | global_id={batchId[0:%3d]classId[1:%3d][2:%zd]} local_id={[0:%zd][1:%zd][2:%zd]}\n",
    //        batchId, classId, get_global_id(2), get_local_id(0), get_local_id(1), get_local_id(2));

    const int loc_label = ((SHARE_LOCATION)? 0 : classId);
    const int scores_size_offset = batchId * NUM_CLASSES + classId;

    buffer3[scores_size_offset] = 0;
    __global BBOXES_INFO *bboxesList = (__global BBOXES_INFO*)&buffer0[(batchId * NUM_LOC_CLASSES + loc_label) * BUFFER_STRIDE];
    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer1[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];

    for (uint idx_prior = 0; idx_prior < NUM_OF_PRIORS; idx_prior++)
    {
        if (SHARE_LOCATION) {
            if (classId == loc_label) {
                UNIT_TYPE decoded_bbox[4];
                FUNC_CALL(get_decoded_bbox)(decoded_bbox, input_location, input_prior_box, idx_prior, classId, batchId);
                BBOXES_INFO bbox_info;
                bbox_info.xmin = decoded_bbox[0];
                bbox_info.ymin = decoded_bbox[1];
                bbox_info.xmax = decoded_bbox[2];
                bbox_info.ymax = decoded_bbox[3];
                bboxesList[idx_prior] = bbox_info;
            }
        } else {
            if (loc_label != BACKGROUND_LABEL_ID) {
                UNIT_TYPE decoded_bbox[4];
                FUNC_CALL(get_decoded_bbox)(decoded_bbox, input_location, input_prior_box, idx_prior, classId, batchId);
                BBOXES_INFO bbox_info;
                bbox_info.xmin = decoded_bbox[0];
                bbox_info.ymin = decoded_bbox[1];
                bbox_info.xmax = decoded_bbox[2];
                bbox_info.ymax = decoded_bbox[3];
                bboxesList[idx_prior] = bbox_info;
            }
        }
        UNIT_TYPE score = FUNC_CALL(get_score)(input_confidence, idx_prior, classId, batchId);
        if (score > 0) {
            int acc_num = buffer3[scores_size_offset];
            SCORES_INFO score_info;
            score_info.batchId = batchId;
            score_info.classId = classId;
            score_info.boxId = idx_prior;
            score_info.score = score;
            scoresList[acc_num] = score_info;
            buffer3[scores_size_offset] = (acc_num + 1);
        }
    }
}
#endif /* IS_ZERO_ITER_CAFFE_OPT */

#ifdef IS_ZERO_ITER_MXNET
KERNEL (detection_output_stage_0_mxnet)(
    __global UNIT_TYPE* input_location,
    __global UNIT_TYPE* input_confidence,
    __global UNIT_TYPE* input_prior_box,
    __global uchar *buffer0,
    __global uchar *buffer1,
    __global int *buffer3)
{
    //printf("detection_output_stage_0_mxnet | NUM_OF_IMAGES=[%3d] NUM_LOC_CLASSES=[%3d]/NUM_CLASSES=[%3d] NUM_OF_PRIORS=[%3d], CONFIDENCE_THRESHOLD=[%f]\n", NUM_OF_IMAGES, NUM_LOC_CLASSES, NUM_CLASSES, NUM_OF_PRIORS, CONFIDENCE_THRESHOLD);

    __global BBOXES_INFO *bboxesList = (__global BBOXES_INFO*)&buffer0[0];
    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer1[0];

    for (uint idx_image = 0; idx_image < NUM_OF_IMAGES; idx_image++)
    {
        buffer3[idx_image] = 0;
        for (uint idx_prior = 0; idx_prior < NUM_OF_PRIORS; idx_prior++)
        {
            for (uint idx_class = 0; idx_class < NUM_LOC_CLASSES; idx_class++)
            {
                int loc_label = ((SHARE_LOCATION)? 0 : idx_class);
                if (!SHARE_LOCATION && loc_label == BACKGROUND_LABEL_ID)
                {
                    continue;
                }
                int bboxes_offset = (idx_image * NUM_LOC_CLASSES * NUM_OF_PRIORS) + idx_class * NUM_OF_PRIORS + idx_prior;
                UNIT_TYPE decoded_bbox[4];
                FUNC_CALL(get_decoded_bbox)(decoded_bbox, input_location, input_prior_box, idx_prior, idx_class, idx_image);
                BBOXES_INFO bbox_info;
                bbox_info.xmin = decoded_bbox[0];
                bbox_info.ymin = decoded_bbox[1];
                bbox_info.xmax = decoded_bbox[2];
                bbox_info.ymax = decoded_bbox[3];
                bboxesList[bboxes_offset] = bbox_info;
                //printf("bboxesList[%d] = [%f, %f, %f, %f]\n", bboxes_offset, bbox_info.xmin, bbox_info.ymin, bbox_info.xmax, bbox_info.ymax);
            }
            int idx_max_score = FUNC_CALL(get_largest_score)(input_confidence, idx_prior, idx_image);
            UNIT_TYPE score = FUNC_CALL(get_score)(input_confidence, idx_prior, idx_max_score, idx_image);
            if (score > 0) {
                int acc_num = buffer3[idx_image];
                int scores_offset = (idx_image * NUM_OF_PRIORS) + acc_num;
                SCORES_INFO score_info;
                score_info.batchId = idx_image;
                score_info.classId = idx_max_score;
                score_info.boxId = idx_prior;
                score_info.score = score;
                scoresList[scores_offset] = score_info;
                buffer3[idx_image] = (acc_num + 1);
                //printf("detection_output_stage_0_mxnet | scoresList[%d] = [batchId:%d, classId:%d, boxId:%d, score:%f]\n", scores_offset, score_info.batchId, score_info.classId, score_info.boxId, score_info.score);
            }
        }

    }
}
#endif /* IS_ZERO_ITER_MXNET */

#ifdef IS_ZERO_ITER_MXNET_OPT
KERNEL (detection_output_stage_0_mxnet_opt)(
    __global UNIT_TYPE* input_location,
    __global UNIT_TYPE* input_confidence,
    __global UNIT_TYPE* input_prior_box,
    __global uchar *buffer0,
    __global uchar *buffer1,
    volatile __global int *buffer3)
{
    const int batchId = get_global_id(0);
    const int priorId = get_global_id(1);
    //printf("detection_output_stage_0_mxnet_opt | global_id={batchId[0:%3d]priorId[1:%3d][2:%zd]} local_id={[0:%zd][1:%zd][2:%zd]}\n",
    //        batchId, priorId, get_global_id(2), get_local_id(0), get_local_id(1), get_local_id(2));

    const int scores_size_offset = batchId * NUM_OF_PRIORS + priorId;
    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer1[batchId * BUFFER_STRIDE];

    if (priorId == 0)
    {
        buffer3[batchId] = 0;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (uint idx_class = 0; idx_class < NUM_LOC_CLASSES; idx_class++)
    {
        const int loc_label = ((SHARE_LOCATION)? 0 : idx_class);
        __global BBOXES_INFO *bboxesList = (__global BBOXES_INFO*)&buffer0[(batchId * NUM_LOC_CLASSES + loc_label) * BUFFER_STRIDE];

        if (SHARE_LOCATION) {
            if (idx_class == loc_label) {
                UNIT_TYPE decoded_bbox[4];
                FUNC_CALL(get_decoded_bbox)(decoded_bbox, input_location, input_prior_box, priorId, idx_class, batchId);
                BBOXES_INFO bbox_info;
                bbox_info.xmin = decoded_bbox[0];
                bbox_info.ymin = decoded_bbox[1];
                bbox_info.xmax = decoded_bbox[2];
                bbox_info.ymax = decoded_bbox[3];
                bboxesList[priorId] = bbox_info;
            }
        } else {
            if (loc_label != BACKGROUND_LABEL_ID) {
                UNIT_TYPE decoded_bbox[4];
                FUNC_CALL(get_decoded_bbox)(decoded_bbox, input_location, input_prior_box, priorId, idx_class, batchId);
                BBOXES_INFO bbox_info;
                bbox_info.xmin = decoded_bbox[0];
                bbox_info.ymin = decoded_bbox[1];
                bbox_info.xmax = decoded_bbox[2];
                bbox_info.ymax = decoded_bbox[3];
                bboxesList[priorId] = bbox_info;
            }
        }
    }
    int idx_max_score = FUNC_CALL(get_largest_score)(input_confidence, priorId, batchId);
    UNIT_TYPE score = FUNC_CALL(get_score)(input_confidence, priorId, idx_max_score, batchId);
    SCORES_INFO score_info;
    score_info.batchId = batchId;
    score_info.classId = idx_max_score;
    score_info.boxId = priorId;
    score_info.score = score;
    scoresList[priorId] = score_info;
    atomic_inc(&buffer3[batchId]);
}
#endif /* IS_ZERO_ITER_MXNET_OPT */

#ifdef IS_FIRST_ITER_CAFFE
KERNEL (detection_output_stage_1_caffe)(
    __global uchar *buffer1,
    __global int *buffer3)
{
    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer1[0];

    for (uint idx_image = 0; idx_image < NUM_OF_IMAGES; idx_image++)
    {
        for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++)
        {
            int scores_size_offset = idx_image * NUM_CLASSES + idx_class;
            if (idx_class == BACKGROUND_LABEL_ID)
            {
                buffer3[scores_size_offset] = 0;
                continue;
            }
            int acc_num = buffer3[scores_size_offset];
            int scores_offset = (idx_image * NUM_CLASSES * NUM_OF_PRIORS) + idx_class * NUM_OF_PRIORS;
            FUNC_CALL(quickSortIterative)(&scoresList[scores_offset], 0, acc_num - 1, true);
            if (TOP_K != -1 && TOP_K < acc_num)
            {
                buffer3[scores_size_offset] = TOP_K;
            }
            //printf("detection_output_stage_1 | acc_num[%d] TOP_K[%3d]\n", acc_num, TOP_K);
            //for (uint idx_prior = 0; idx_prior < acc_num; idx_prior++)
            //{
            //    SCORES_INFO score_info;
            //    score_info = scoresList[scores_offset + idx_prior];
            //    printf("detection_output_stage_1 | scoresList[%d] = [batchId:%d, classId:%d, boxId:%d, score:%f]\n", scores_offset + idx_prior, score_info.batchId, score_info.classId, score_info.boxId, score_info.score);
            //}
        }
    }
}
#endif /* IS_FIRST_ITER_CAFFE */

#ifdef IS_FIRST_ITER_CAFFE_OPT
KERNEL (detection_output_stage_1_caffe_opt)(
    __global uchar *buffer1,
    __global int *buffer3)
{
    const int batchId = get_global_id(0);
    const int classId = get_global_id(1);
    const int workItemId = get_global_id(2);
    const int localClassId = get_local_id(1);
    __local int __range[LOCAL_CLASS_NUM][LOCAL_WORK_NUM * 4];

    //printf("detection_output_stage_1 |  global_id={batchId[0:%3d]classId[1:%3d]workItemId[2:%3d]} local_id={[0:%zd]localClassId[1:%3d][2:%zd]}\n",
    //        batchId, classId, workItemId, get_local_id(0), localClassId, get_local_id(2));

    const int scoresInfoNum = buffer3[batchId * NUM_CLASSES + classId];
    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer1[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];

    if (classId == BACKGROUND_LABEL_ID) {
        if (workItemId == 0) {
            buffer3[batchId * NUM_CLASSES + classId] = 0;
        }
    } else {
        //if (batchId == 0 && classId == 1 && workItemId == 0) {
        //    printf("detection_output_stage_1 | buffer1 idx=[(batchId(%d) * NUM_CLASSES(%d) + classId(%d)) * BUFFER_STRIDE(%d)]=%d\n",
        //            batchId, NUM_CLASSES, classId, BUFFER_STRIDE, (batchId * NUM_CLASSES + classId) * BUFFER_STRIDE);
        //    printf("detection_output_stage_1 | scoresInfoNum[%d] scoresList = { ", scoresInfoNum);
        //    for (uint idx_score_info = 0; idx_score_info < scoresInfoNum; idx_score_info++)
        //    {
        //        SCORES_INFO score_info;
        //        score_info = scoresList[idx_score_info];
        //        printf("[%f]", score_info.score);
        //    }
        //    printf("}\n");
        //}
        if (workItemId == 0 && scoresInfoNum > 1) {
            __range[localClassId][0] = 0;
            __range[localClassId][1] = scoresInfoNum - 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int range_step = 2;
        const int first_id = workItemId * 2;
        for (int i = 0; i < PARTITION_STEP; ++i, range_step *= 2) {
            if (scoresInfoNum > 1 && workItemId <= i) {
                const int begin_id = __range[localClassId][first_id];
                const int end_id = __range[localClassId][first_id + 1];
                const int second_id = first_id + range_step;

                if (begin_id < end_id) {
                    const int pivot = FUNC_CALL(partition)(scoresList, begin_id, end_id, true);
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

        if (scoresInfoNum > 1) {
            const int begin_id = __range[localClassId][first_id];
            const int end_id = __range[localClassId][first_id + 1];
            if (begin_id < end_id) {
                FUNC_CALL(quickSortIterative)(scoresList, begin_id, end_id, true);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (workItemId == 0 && (TOP_K != -1 && TOP_K < scoresInfoNum)) {
            buffer3[batchId * NUM_CLASSES + classId] = TOP_K;
        }
    }
}
#endif /* IS_FIRST_ITER_CAFFE_OPT */

#ifdef IS_FIRST_ITER_MXNET
KERNEL (detection_output_stage_1_mxnet)(
    __global uchar *buffer1,
    __global int *buffer3)
{
    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer1[0];

    for (uint idx_image = 0; idx_image < NUM_OF_IMAGES; idx_image++)
    {
        int acc_num = buffer3[idx_image];
        int scores_offset = idx_image * NUM_OF_PRIORS;
        FUNC_CALL(quickSortIterative)(&scoresList[scores_offset], 0, acc_num - 1, false);
        if (TOP_K != -1 && TOP_K < acc_num)
        {
            buffer3[idx_image] = TOP_K;
        }
        //printf("detection_output_stage_1_mxnet | acc_num[%d] TOP_K[%3d]\n", acc_num, TOP_K);
        //for (uint idx_score = 0; idx_score < acc_num; idx_score++)
        //{
        //    SCORES_INFO score_info;
        //    score_info = scoresList[scores_offset + idx_score];
        //    printf("detection_output_stage_1_mxnet | scoresList[%d] = [batchId:%d, classId:%d, boxId:%d, score:%f]\n", scores_offset + idx_score, score_info.batchId, score_info.classId, score_info.boxId, score_info.score);
        //}
    }
}
#endif /* IS_FIRST_ITER_MXNET */

#ifdef IS_FIRST_ITER_MXNET_OPT
KERNEL (detection_output_stage_1_mxnet_opt)(
    __global uchar *buffer1,
    __global int *buffer3)
{
    const int batchId = get_global_id(0);
    const int workItemId = get_global_id(2);
    __local int __range[LOCAL_WORK_NUM * 4];

    //printf("detection_output_stage_1_mxnet | global_id={batchId[0:%d][1:%d]workItemId[2:%zd]} local_id={[0:%zd][1:%zd][2:%zd]}\n", batchId, get_global_id(1), workItemId, get_local_id(0), get_local_id(1), get_local_id(2));

    const int scoresInfoNum = buffer3[batchId];
    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer1[batchId * BUFFER_STRIDE];

    if (workItemId == 0 && scoresInfoNum > 1) {
        __range[0] = 0;
        __range[1] = scoresInfoNum - 1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int range_step = 2;
    const int first_id = workItemId * 2;
    for (int i = 0; i < PARTITION_STEP; ++i, range_step *= 2) {
        if (scoresInfoNum > 1 && workItemId <= i) {
            const int begin_id = __range[first_id];
            const int end_id = __range[first_id + 1];
            const int second_id = first_id + range_step;
            if (begin_id < end_id) {
                const int pivot = FUNC_CALL(partition)(scoresList, begin_id, end_id, true);
                __range[first_id     ] = begin_id;
                __range[first_id + 1 ] = max(pivot - 1, begin_id);
                __range[second_id    ] = min(pivot + 1, end_id);
                __range[second_id + 1] = end_id;
            } else {
                __range[second_id    ] = 0;
                __range[second_id + 1] = 0;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (scoresInfoNum > 1) {
        const int begin_id = __range[first_id];
        const int end_id = __range[first_id + 1];
        if (begin_id < end_id) {
            FUNC_CALL(quickSortIterative)(scoresList, begin_id, end_id, true);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (workItemId == 0 && (TOP_K != -1 && TOP_K < scoresInfoNum)) {
        buffer3[batchId] = TOP_K;
    }
}
#endif /* IS_FIRST_ITER_MXNET_OPT */

#ifdef IS_SECOND_ITER_CAFFE
KERNEL (detection_output_stage_2_caffe)(
    __global uchar *buffer0,
    __global uchar *buffer1,
    __global uchar *buffer2,
    __global int *buffer3)
{
    __global BBOXES_INFO *bboxesList = (__global BBOXES_INFO*)&buffer0[0];
    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer1[0];
    __global SCORES_INFO *selectedScoresList = (__global SCORES_INFO*)&buffer2[0];

    __local uint indices[NUM_OF_PRIORS];

    for (uint idx_image = 0; idx_image < NUM_OF_IMAGES; idx_image++)
    {
        for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++)
        {
            if (idx_class == BACKGROUND_LABEL_ID)
            {
                continue;
            }
            int selectedBoxNum = 0;
            int scores_size_offset = idx_image * NUM_CLASSES + idx_class;
            int acc_num = buffer3[scores_size_offset];
            int loc_label = ((SHARE_LOCATION)? 0 : idx_class);
            int bboxes_offset = (idx_image * NUM_LOC_CLASSES * NUM_OF_PRIORS) + loc_label * NUM_OF_PRIORS;
            int scores_offset = (idx_image * NUM_CLASSES * NUM_OF_PRIORS) + idx_class * NUM_OF_PRIORS;

            //printf("detection_output_stage_2 |                 test_overlap1=[%f] vs test_overlap2=[%f]\n",
            //        FUNC_CALL(jaccardOverlap)(&bboxesList[bboxes_offset + 5], &bboxesList[bboxes_offset + 7]),
            //        FUNC_CALL(jaccardOverlap)(&bboxesList[bboxes_offset + 7], &bboxesList[bboxes_offset + 5]));
            //printf("detection_output_stage_2 | acc_num[%d], bboxes_offset[%d]\n", acc_num, bboxes_offset);
            for (uint idx_score = 0; idx_score < acc_num; idx_score++)
            {
                bool keep = true;
                int idx = scoresList[scores_offset + idx_score].boxId;
                //printf("detection_output_stage_2 | ==== idx_score[%d]/idx=[%d], selectedBoxNum=[%d]\n", idx_score, idx, selectedBoxNum);
                for (uint idx_indice = 0; idx_indice < selectedBoxNum; idx_indice++)
                {
                    int kept_idx = indices[idx_indice];
                    float overlap = FUNC_CALL(jaccardOverlap)(&bboxesList[bboxes_offset + idx], &bboxesList[bboxes_offset + kept_idx]);
                    //printf("detection_output_stage_2 |     ==== idx=[%d] and kept_idx=[%d] => overlap(%f)/NMS_THRESHOLD=(%f)\n", idx, kept_idx, overlap, NMS_THRESHOLD);
                    if (overlap > NMS_THRESHOLD)
                    {
                        keep = false;
                        break;
                    }
                }
                if (keep)
                {
                    SCORES_INFO score_info;
                    score_info.batchId = scoresList[scores_offset + idx_score].batchId;
                    score_info.classId = scoresList[scores_offset + idx_score].classId;
                    score_info.boxId = scoresList[scores_offset + idx_score].boxId;
                    score_info.score = scoresList[scores_offset + idx_score].score;
                    selectedScoresList[scores_offset + selectedBoxNum] = score_info;
                    //printf("detection_output_stage_2 | ==== keep!!! idx=[%d]\n", idx);
                    indices[selectedBoxNum] = idx;
                    ++selectedBoxNum;
                }
            }
            //printf("detection_output_stage_2 | buffer3[%d] = %d\n", scores_size_offset, selectedBoxNum);
            buffer3[scores_size_offset] = selectedBoxNum;
        }
    }
}
#endif /* IS_SECOND_ITER_CAFFE */

#ifdef IS_SECOND_ITER_CAFFE_OPT
KERNEL (detection_output_stage_2_caffe_opt)(
    __global uchar *buffer0,
    __global uchar *buffer1,
    __global uchar *buffer2,
    __global int *buffer3)
{
    const int batchId = get_global_id(0);
    const int classId = get_global_id(1);
    const int loc_label = ((SHARE_LOCATION)? 0 : classId);
    const int scoresInfoIdx = batchId * NUM_CLASSES + classId;

    __global BBOXES_INFO *bboxesList = (__global BBOXES_INFO*)&buffer0[(batchId * NUM_LOC_CLASSES + loc_label) * BUFFER_STRIDE];
    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer1[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];
    __global SCORES_INFO *selectedScoresList = (__global SCORES_INFO*)&buffer2[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];

    const int scoresInfoNum = buffer3[scoresInfoIdx];
    __local uint indices[NUM_OF_PRIORS];

    //printf("detection_output_stage_2_caffe_opt | global_id={batchId[0:%3d]classId[1:%3d][2:%zd]} local_id={[0:%zd][1:%zd][2:%zd]} scoresInfoNum=[%d]\n",
    //        batchId, classId, get_global_id(2), get_local_id(0), get_local_id(1), get_local_id(2), scoresInfoNum);

    int selectedBoxNum = 0;
    for (uint idx_score = 0; idx_score < scoresInfoNum; idx_score++)
    {
        bool keep = true;
        int idx = scoresList[idx_score].boxId;
        for (uint idx_indice = 0; idx_indice < selectedBoxNum; idx_indice++)
        {
            int kept_idx = indices[idx_indice];
            float overlap = FUNC_CALL(jaccardOverlap)(&bboxesList[idx], &bboxesList[kept_idx]);
            if (overlap > NMS_THRESHOLD)
            {
                keep = false;
                break;
            }
        }
        if (keep)
        {
            SCORES_INFO score_info;
            score_info.batchId = scoresList[idx_score].batchId;
            score_info.classId = scoresList[idx_score].classId;
            score_info.boxId = scoresList[idx_score].boxId;
            score_info.score = scoresList[idx_score].score;
            selectedScoresList[selectedBoxNum] = score_info;
            indices[selectedBoxNum] = idx;
            ++selectedBoxNum;
        }
    }
    buffer3[scoresInfoIdx] = selectedBoxNum;
}
#endif /* IS_SECOND_ITER_CAFFE_OPT */

#ifdef IS_SECOND_ITER_MXNET
KERNEL (detection_output_stage_2_mxnet)(
    __global uchar *buffer0,
    __global uchar *buffer1,
    __global uchar *buffer2,
    __global int *buffer3)
{
    __global BBOXES_INFO *bboxesList = (__global BBOXES_INFO*)&buffer0[0];
    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer1[0];
    __global SCORES_INFO *selectedScoresList = (__global SCORES_INFO*)&buffer2[0];

    __local int indices[NUM_CLASSES * NUM_OF_PRIORS];
    __local int num_indice[NUM_CLASSES];

    for (uint idx_image = 0; idx_image < NUM_OF_IMAGES; idx_image++)
    {
        for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++)
        {
            num_indice[idx_class] = 0;
        }
        int selectedBoxNum = 0;
        int acc_num = buffer3[idx_image];
        int scores_offset = idx_image * NUM_OF_PRIORS;

        for (uint idx_score = 0; idx_score < acc_num; idx_score++)
        {
            bool keep = true;
            int idx = scoresList[scores_offset + idx_score].boxId;
            int cls = scoresList[scores_offset + idx_score].classId;
            int loc_label = ((SHARE_LOCATION)? 0 : cls);
            int bboxes_offset = (idx_image * NUM_LOC_CLASSES * NUM_OF_PRIORS) + loc_label * NUM_OF_PRIORS;
            int indice_offset = cls * NUM_OF_PRIORS;
            int cur_num_indice = num_indice[cls];
            for (uint idx_indice = 0; idx_indice < num_indice[cls]; idx_indice++)
            {
                int kept_idx = indices[indice_offset + idx_indice];
                float overlap = FUNC_CALL(jaccardOverlap)(&bboxesList[bboxes_offset + idx], &bboxesList[bboxes_offset + kept_idx]);
                //printf("detection_output_stage_2_mxnet |     ==== idx=[%d] and kept_idx=[%d] => overlap(%f)/NMS_THRESHOLD=(%f)\n", idx, kept_idx, overlap, NMS_THRESHOLD);
                if (overlap > NMS_THRESHOLD)
                {
                    keep = false;
                    break;
                }
            }
            if (keep)
            {
                SCORES_INFO score_info;
                score_info.batchId = scoresList[scores_offset + idx_score].batchId;
                score_info.classId = scoresList[scores_offset + idx_score].classId;
                score_info.boxId = scoresList[scores_offset + idx_score].boxId;
                score_info.score = scoresList[scores_offset + idx_score].score;
                selectedScoresList[scores_offset + selectedBoxNum] = score_info;
                //printf("detection_output_stage_2_mxnet | keep idx=[%d] selectedScoresList[%d] = [batchId:%d, classId:%d, boxId:%d, score:%f]\n", idx, scores_offset + selectedBoxNum, score_info.batchId, score_info.classId, score_info.boxId, score_info.score);
                num_indice[cls] = cur_num_indice + 1;
                indices[indice_offset + cur_num_indice] = idx;
                ++selectedBoxNum;
            }
        }
        buffer3[idx_image] = selectedBoxNum;
        //printf("detection_output_stage_2_mxnet | buffer3[%d] = [%d]\n", idx_image, selectedBoxNum);
    }
}
#endif /* IS_SECOND_ITER_MXNET */

#ifdef IS_SECOND_ITER_MXNET_OPT
KERNEL (detection_output_stage_2_mxnet_opt)(
    __global uchar *buffer0,
    __global uchar *buffer1,
    __global uchar *buffer2,
    __global int *buffer3)
{
    const int batchId = get_global_id(0);
    const int scoresInfoNum = buffer3[batchId];

    //printf("detection_output_stage_2_mxnet | global_id={batchId[0:%3d][1:%zd][2:%zd]} local_id={[0:%zd][1:%zd][2:%zd]}\n",
    //        batchId, get_global_id(1), get_global_id(2), get_local_id(0), get_local_id(1), get_local_id(2));

    __global BBOXES_INFO *bboxesList = (__global BBOXES_INFO*)&buffer0[0];
    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer1[batchId * BUFFER_STRIDE];
    __global SCORES_INFO *selectedScoresList = (__global SCORES_INFO*)&buffer2[batchId * BUFFER_STRIDE];

    __local int indices[NUM_CLASSES * NUM_OF_PRIORS];
    __local int num_indice[NUM_CLASSES];

    for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++)
    {
        num_indice[idx_class] = 0;
    }

    int selectedBoxNum = 0;
    for (uint idx_score = 0; idx_score < scoresInfoNum; idx_score++)
    {
        bool keep = true;
        int idx = scoresList[idx_score].boxId;
        int cls = scoresList[idx_score].classId;
        int loc_label = ((SHARE_LOCATION)? 0 : cls);
        int bboxes_offset = (batchId * NUM_LOC_CLASSES * NUM_OF_PRIORS) + loc_label * NUM_OF_PRIORS;
        int indice_offset = cls * NUM_OF_PRIORS;
        int cur_num_indice = num_indice[cls];
        for (uint idx_indice = 0; idx_indice < cur_num_indice; idx_indice++)
        {
            int kept_idx = indices[indice_offset + idx_indice];
            float overlap = FUNC_CALL(jaccardOverlap)(&bboxesList[bboxes_offset + idx], &bboxesList[bboxes_offset + kept_idx]);
            if (overlap > NMS_THRESHOLD)
            {
                keep = false;
                break;
            }
        }
        if (keep)
        {
            SCORES_INFO score_info;
            score_info.batchId = scoresList[idx_score].batchId;
            score_info.classId = scoresList[idx_score].classId;
            score_info.boxId = scoresList[idx_score].boxId;
            score_info.score = scoresList[idx_score].score;
            selectedScoresList[selectedBoxNum] = score_info;
            num_indice[cls] = cur_num_indice + 1;
            indices[indice_offset + cur_num_indice] = idx;
            ++selectedBoxNum;
        }
    }
    buffer3[batchId] = selectedBoxNum;
}
#endif /* IS_SECOND_ITER_MXNET_OPT */

#ifdef IS_THIRD_ITER_CAFFE
KERNEL (detection_output_stage_final_caffe)(
    __global UNIT_TYPE* output,
    __global uchar *buffer0,
    __global uchar *buffer1,
    __global uchar *buffer2,
    __global int *buffer3)
{
    __global BBOXES_INFO *bboxesList = (__global BBOXES_INFO*)&buffer0[0];
    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer1[0];
    __global SCORES_INFO *selectedScoresList = (__global SCORES_INFO*)&buffer2[0];

    for (uint idx_image = 0; idx_image < NUM_OF_IMAGES; idx_image++)
    {
        int num_det = 0;
        for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++)
        {
            if (idx_class == BACKGROUND_LABEL_ID)
            {
                continue;
            }
            int scores_size_offset = idx_image * NUM_CLASSES + idx_class;
            const int acc_num = buffer3[scores_size_offset];
            int scores_offset = (idx_image * NUM_CLASSES * NUM_OF_PRIORS) + idx_class * NUM_OF_PRIORS;
            //printf("detection_output_stage_final | acc_num[%d]\n", acc_num);
            for (uint idx_score = 0; idx_score < acc_num; idx_score++)
            {
                SCORES_INFO score_info;
                score_info = selectedScoresList[scores_offset + idx_score];
                scoresList[num_det + idx_score] = score_info;
                //printf("detection_output_stage_final | selectedScoresList[%d] = [batchId:%d, classId:%d, boxId:%d, score:%f] -> %d\n", scores_offset + idx_score, score_info.batchId, score_info.classId, score_info.boxId, score_info.score, num_det + idx_score);
            }
            num_det += acc_num;
            buffer3[scores_size_offset] = 0;
        }
        //printf("detection_output_stage_final | num_det[%d] before checking keep_top_k\n", num_det);

        FUNC_CALL(quickSortIterative)(scoresList, 0, num_det - 1, true);

        if (KEEP_TOP_K > -1 && num_det > KEEP_TOP_K)
        {
            num_det = KEEP_TOP_K;
        }
        //printf("detection_output_stage_final | num_det[%d] after checking keep_top_k\n", num_det);
        for (uint idx_num_det = 0; idx_num_det < num_det; idx_num_det++)
        {
            SCORES_INFO score_info;
            score_info = scoresList[idx_num_det];
            //printf("detection_output_stage_final | scoresList[%d] = [batchId:%d, classId:%d, boxId:%d, score:%f]\n", idx_num_det, score_info.batchId, score_info.classId, score_info.boxId, score_info.score);
            int scores_size_offset = idx_image * NUM_CLASSES + score_info.classId;
            int acc_num = buffer3[scores_size_offset];
            int scores_offset = (idx_image * NUM_CLASSES * NUM_OF_PRIORS) + score_info.classId * NUM_OF_PRIORS + acc_num;
            selectedScoresList[scores_offset] = score_info;
            buffer3[scores_size_offset] = (acc_num + 1);
        }

        //for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++)
        //{
        //    if (idx_class == BACKGROUND_LABEL_ID)
        //    {
        //        continue;
        //    }
        //    int scores_size_offset = idx_image * NUM_CLASSES + idx_class;
        //    int acc_num = buffer3[scores_size_offset];
        //    int scores_offset = (idx_image * NUM_CLASSES * NUM_OF_PRIORS) + idx_class * NUM_OF_PRIORS;
        //    printf("detection_output_stage_final | acc_num[%d]\n", acc_num);
        //    for (uint idx_score = 0; idx_score < acc_num; idx_score++)
        //    {
        //        SCORES_INFO score_info;
        //        score_info = selectedScoresList[scores_offset + idx_score];
        //        printf("detection_output_stage_final | selectedScoresList[%d] = [batchId:%d, classId:%d, boxId:%d, score:%f]\n", scores_offset + idx_score, score_info.batchId, score_info.classId, score_info.boxId, score_info.score);
        //    }
        //}
    }
    //printf("gpu kernel result =====================================\n");
    int count = 0;
    for (uint idx_image = 0; idx_image < NUM_OF_IMAGES; idx_image++)
    {
        for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++)
        {
            int scores_size_offset = idx_image * NUM_CLASSES + idx_class;
            int acc_num = buffer3[scores_size_offset];
            if (acc_num == 0)
            {
                continue;
            }
            int scores_offset = (idx_image * NUM_CLASSES * NUM_OF_PRIORS) + idx_class * NUM_OF_PRIORS;
            int loc_label = ((SHARE_LOCATION)? 0 : idx_class);
            int bboxes_offset = (idx_image * NUM_LOC_CLASSES * NUM_OF_PRIORS) + loc_label * NUM_OF_PRIORS;
            for (uint idx_score = 0; idx_score < acc_num; idx_score++)
            {
                SCORES_INFO score_info;
                score_info = selectedScoresList[scores_offset + idx_score];
                output[count * OUTPUT_ROW_SIZE] = score_info.batchId;
                output[count * OUTPUT_ROW_SIZE + 1] = ((DECREASE_LABEL_ID) ? score_info.classId - 1 : score_info.classId);
                output[count * OUTPUT_ROW_SIZE + 2] = score_info.score;
                BBOXES_INFO bbox_info;
                bbox_info = bboxesList[bboxes_offset + score_info.boxId];
                float xmin = bbox_info.xmin;
                float ymin = bbox_info.ymin;
                float xmax = bbox_info.xmax;
                float ymax = bbox_info.ymax;

                if (CLIP_AFTER_NMS) {
                    xmin = max(0.0f, min(1.0f, xmin));
                    ymin = max(0.0f, min(1.0f, ymin));
                    xmax = max(0.0f, min(1.0f, xmax));
                    ymax = max(0.0f, min(1.0f, ymax));
                }
                output[count * OUTPUT_ROW_SIZE + 3] = xmin;
                output[count * OUTPUT_ROW_SIZE + 4] = ymin;
                output[count * OUTPUT_ROW_SIZE + 5] = xmax;
                output[count * OUTPUT_ROW_SIZE + 6] = ymax;
                //printf("[%d, %d, %f, %f, %f, %f, %f] -> [%d]\n", score_info.batchId, score_info.classId, score_info.score, xmin, ymin, xmax, ymax, score_info.boxId);
                ++count;
            }
        }
    }

    if (count < NUM_OF_IMAGES * KEEP_TOP_K)
    {
        output[count * OUTPUT_ROW_SIZE] = -1.f;
        //printf("[-1.0, , , , , , ]\n");
    }
    //printf("===============================================\n");
}
#endif  /* IS_THIRD_ITER_CAFFE */

#ifdef IS_THIRD_ITER_MXNET
KERNEL (detection_output_stage_final_mxnet)(
    __global UNIT_TYPE* output,
    __global uchar *buffer0,
    __global uchar *buffer1,
    __global uchar *buffer2,
    __global int *buffer3)
{
    __global BBOXES_INFO *bboxesList = (__global BBOXES_INFO*)&buffer0[0];
    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer1[0];
    __global SCORES_INFO *selectedScoresList = (__global SCORES_INFO*)&buffer2[0];

    __local int num_of_det[NUM_OF_IMAGES * NUM_CLASSES];

    for (uint idx_image = 0; idx_image < NUM_OF_IMAGES; idx_image++)
    {
        for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++)
        {
            int scores_size_offset = idx_image * NUM_CLASSES + idx_class;
            num_of_det[scores_size_offset] = 0;
        }
    }
    for (uint idx_image = 0; idx_image < NUM_OF_IMAGES; idx_image++)
    {
        int num_det = 0;
        num_det = buffer3[idx_image];
        int scores_offset = idx_image * NUM_OF_PRIORS;
        if (KEEP_TOP_K > -1 && num_det > KEEP_TOP_K)
        {
            FUNC_CALL(quickSortIterative)(&selectedScoresList[scores_offset], 0, num_det - 1, true);
            num_det = KEEP_TOP_K;
        }
        for (uint idx_num_det = 0; idx_num_det < num_det; idx_num_det++)
        {
            SCORES_INFO score_info;
            score_info = selectedScoresList[scores_offset + idx_num_det];
            //printf("detection_output_stage_final_mxnet | selectedScoresList[%d] = [batchId:%d, classId:%d, boxId:%d, score:%f]\n", idx_num_det, score_info.batchId, score_info.classId, score_info.boxId, score_info.score);
            int scores_size_offset = idx_image * NUM_CLASSES + score_info.classId;
            int acc_num = num_of_det[scores_size_offset];
            int scores_offset = (idx_image * NUM_CLASSES * NUM_OF_PRIORS) + score_info.classId * NUM_OF_PRIORS + acc_num;
            scoresList[scores_offset] = score_info;
            num_of_det[scores_size_offset] = (acc_num + 1);
            //printf("detection_output_stage_final_mxnet | ==> scoresList[%d] = [batchId:%d, classId:%d, boxId:%d, score:%f]\n", scores_offset, score_info.batchId, score_info.classId, score_info.boxId, score_info.score);
            //printf("detection_output_stage_final_mxnet | ==> num_of_det[%d] = [%d]\n", scores_size_offset, (acc_num + 1));
        }
    }
    //printf("gpu kernel result =====================================\n");
    int count = 0;
    for (uint idx_image = 0; idx_image < NUM_OF_IMAGES; idx_image++)
    {
        for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++)
        {
            int scores_size_offset = idx_image * NUM_CLASSES + idx_class;
            int acc_num = num_of_det[scores_size_offset];
            //printf("detection_output_stage_final_mxnet | idx_image[%d] idx_class[%d] num_of_det[%d] = [%d]\n", idx_image, idx_class, scores_size_offset, acc_num);
            if (acc_num == 0)
            {
                continue;
            }
            int scores_offset = (idx_image * NUM_CLASSES * NUM_OF_PRIORS) + idx_class * NUM_OF_PRIORS;
            int loc_label = ((SHARE_LOCATION)? 0 : idx_class);
            int bboxes_offset = (idx_image * NUM_LOC_CLASSES * NUM_OF_PRIORS) + loc_label * NUM_OF_PRIORS;
            for (uint idx_score = 0; idx_score < acc_num; idx_score++)
            {
                SCORES_INFO score_info;
                score_info = scoresList[scores_offset + idx_score];
                output[count * OUTPUT_ROW_SIZE] = score_info.batchId;
                output[count * OUTPUT_ROW_SIZE + 1] = ((DECREASE_LABEL_ID) ? score_info.classId - 1 : score_info.classId);
                output[count * OUTPUT_ROW_SIZE + 2] = score_info.score;
                BBOXES_INFO bbox_info;
                bbox_info = bboxesList[bboxes_offset + score_info.boxId];
                float xmin = bbox_info.xmin;
                float ymin = bbox_info.ymin;
                float xmax = bbox_info.xmax;
                float ymax = bbox_info.ymax;

                if (CLIP_AFTER_NMS) {
                    xmin = max(0.0f, min(1.0f, xmin));
                    ymin = max(0.0f, min(1.0f, ymin));
                    xmax = max(0.0f, min(1.0f, xmax));
                    ymax = max(0.0f, min(1.0f, ymax));
                }
                output[count * OUTPUT_ROW_SIZE + 3] = xmin;
                output[count * OUTPUT_ROW_SIZE + 4] = ymin;
                output[count * OUTPUT_ROW_SIZE + 5] = xmax;
                output[count * OUTPUT_ROW_SIZE + 6] = ymax;
                //printf("[%d, %d, %f, %f, %f, %f, %f] -> [%d]\n", score_info.batchId, ((DECREASE_LABEL_ID) ? score_info.classId - 1 : score_info.classId), score_info.score, xmin, ymin, xmax, ymax, score_info.boxId);
                ++count;
            }
        }
    }

    if (count < NUM_OF_IMAGES * KEEP_TOP_K)
    {
        output[count * OUTPUT_ROW_SIZE] = -1.f;
        //printf("[-1.0, , , , , , ]\n");
    }
    //printf("===============================================\n");
}
#endif  /* IS_THIRD_ITER_MXNET */
