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

inline int FUNC(partition)(__global SCORES_INFO* arr, int l, int h)
{
    UNIT_TYPE x = arr[h].score;
    int i = (l - 1);
    for (int j = l; j <= h - 1; j++) {
        if (arr[j].score >= x) {
            i++;
            FUNC_CALL(swap_scores_info)(&arr[i], &arr[j]);
        }
    }
    FUNC_CALL(swap_scores_info)(&arr[i + 1], &arr[h]);
    return (i + 1);
}

inline void FUNC(quickSortIterative)(__global SCORES_INFO* arr, int l, int h)
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

#ifdef IS_ZERO_ITER
KERNEL (detection_output_stage_0)(
    __global UNIT_TYPE* input_location,
    __global UNIT_TYPE* input_confidence,
    __global UNIT_TYPE* input_prior_box,
    __global uchar *buffer0,
    __global uchar *buffer1,
    __global int *buffer3)
{
    printf(" detection_output_stage_0 | NUM_OF_IMAGES=[%3d] NUM_LOC_CLASSES=[%3d]/NUM_CLASSES=[%3d] NUM_OF_PRIORS=[%3d], CONFIDENCE_THRESHOLD=[%f]\n", NUM_OF_IMAGES, NUM_LOC_CLASSES, NUM_CLASSES, NUM_OF_PRIORS, CONFIDENCE_THRESHOLD);

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
                int bboxes_offset = (idx_image * NUM_LOC_CLASSES * NUM_OF_PRIORS) + NUM_LOC_CLASSES * idx_class + idx_prior;
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
        //    printf("buffer3[%d] = [%3d]\n", scores_size_offset, buffer3[scores_size_offset]);
        //}
    }
}
#endif /* IS_ZERO_ITER */

#ifdef IS_FIRST_ITER
KERNEL (detection_output_stage_1)(
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
            FUNC_CALL(quickSortIterative)(&scoresList[scores_offset], 0, acc_num - 1);
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
#endif /* IS_FIRST_ITER */

#ifdef IS_SECOND_ITER
KERNEL (detection_output_stage_2)(
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
            int bboxes_offset = (idx_image * NUM_LOC_CLASSES * NUM_OF_PRIORS) + NUM_LOC_CLASSES * loc_label;
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
#endif /* IS_SECOND_ITER */

#ifdef IS_THIRD_ITER
KERNEL (detection_output_stage_final)(
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

        FUNC_CALL(quickSortIterative)(scoresList, 0, num_det - 1);

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
        //        printf("detection_output_stage_1 | scoresList[%d] = [batchId:%d, classId:%d, boxId:%d, score:%f]\n", scores_offset + idx_score, score_info.batchId, score_info.classId, score_info.boxId, score_info.score);
        //    }
        //}
    }
    printf("gpu kernel result =====================================\n");
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
            int bboxes_offset = (idx_image * NUM_LOC_CLASSES * NUM_OF_PRIORS) + NUM_LOC_CLASSES * loc_label;
            for (uint idx_score = 0; idx_score < acc_num; idx_score++)
            {
                SCORES_INFO score_info;
                score_info = selectedScoresList[scores_offset + idx_score];
                output[count * OUTPUT_ROW_SIZE] = score_info.batchId;
                output[count * OUTPUT_ROW_SIZE + 1] = ((DECREASE_LABEL_ID) ? score_info.classId - 1.0f : score_info.classId);
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
                printf("[%d, %d, %f, %f, %f, %f, %f]\n", score_info.batchId, score_info.classId, score_info.score, xmin, ymin, xmax, ymax);
                ++count;
            }
        }
    }

    while (count < NUM_OF_IMAGES * KEEP_TOP_K)
    {
        output[count * OUTPUT_ROW_SIZE] = -1.f;
        output[count * OUTPUT_ROW_SIZE + 1] = 0.f;
        output[count * OUTPUT_ROW_SIZE + 2] = 0.f;
        output[count * OUTPUT_ROW_SIZE + 3] = 0.f;
        output[count * OUTPUT_ROW_SIZE + 4] = 0.f;
        output[count * OUTPUT_ROW_SIZE + 5] = 0.f;
        output[count * OUTPUT_ROW_SIZE + 6] = 0.f;
        printf("[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n");
        ++count;
    }
    printf("===============================================\n");
}
#endif  /* IS_THIRD_ITER */
