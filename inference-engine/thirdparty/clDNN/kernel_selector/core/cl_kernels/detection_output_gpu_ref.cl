// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/data_types.cl"
#include "include/fetch_data.cl"
#include "include/common.cl"
#include "include/unit_type.cl"
#include "include/detection_output_common.cl"

#define unroll_for __attribute__((opencl_unroll_hint)) for
#define NUM_CLASSES_ACC (NUM_CLASSES + 1)

typedef struct {
    int batchId;
    int classId;
    int boxId;
    UNIT_TYPE score;
} FUNC(Scores);

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

inline void FUNC(bubbleSortIterative)(__global SCORES_INFO* arr, int l, int h)
{
    for (int i = 0; i < h-l; i++) {
        bool swapped = false;
        for (int j = l; j < h-i; j++) {
            if ((arr[j].score > arr[j+1].score) || (arr[j].score == arr[j+1].score && arr[j].boxId < arr[j+1].boxId)) {
                FUNC_CALL(swap_scores_info)(&arr[j], &arr[j+1]);
                swapped = true;
            }
        }

        if (!swapped)
            break;
    }
}

inline void FUNC(quickSortIterative)(__global SCORES_INFO* arr, int l, int h, bool use_custom_comp)
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
        int p = FUNC_CALL(partition)(arr, l, h, use_custom_comp);

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

inline int FUNC(get_accumulated_detections)(__global int* buffer2, int batch_id)
{
    int acc_num = 0;
    for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++)
    {
        acc_num += buffer2[batch_id * NUM_CLASSES_ACC + idx_class];
    }
    return acc_num;
}

inline int FUNC(get_start_idx)(__global int* buffer2, int batch_id)
{
    int start_idx = 0;
    for (uint idx_batch = 0; idx_batch < batch_id; idx_batch++)
    {
        const int num_det = buffer2[idx_batch * NUM_CLASSES_ACC + NUM_CLASSES];
        start_idx += (num_det > KEEP_TOP_K ? KEEP_TOP_K: num_det);
    }
    return start_idx;
}

inline int FUNC(get_final_detections)(__global int* buffer2)
{
    int final_detections = 0;
    for (uint idx_image = 0; idx_image < NUM_OF_IMAGES; idx_image++)
    {
        const int num_det = buffer2[idx_image * NUM_CLASSES_ACC + NUM_CLASSES];
        final_detections += (num_det > KEEP_TOP_K ? KEEP_TOP_K: num_det);
    }
    return final_detections;
}

inline UNIT_TYPE FUNC(jaccardOverlap)(UNIT_TYPE* bbox1, UNIT_TYPE* bbox2)
{
    UNIT_TYPE overlap = 0.0;
    bool intersecting = (bbox1[0] < bbox2[2]) & (bbox2[0] < bbox1[2]) & (bbox1[1] < bbox2[3]) & (bbox2[1] < bbox1[3]);

    if (intersecting)
    {
        const UNIT_TYPE intersect_width = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]);
        const UNIT_TYPE intersect_height = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]);
        if (intersect_width > 0 && intersect_height > 0) {
            const UNIT_TYPE intersect_size = intersect_width * intersect_height;
            const UNIT_TYPE bbox1_size = (bbox1[2] - bbox1[0]) * (bbox1[3]- bbox1[1]);
            const UNIT_TYPE bbox2_size = (bbox2[2] - bbox2[0]) * (bbox2[3]- bbox2[1]);
            overlap = intersect_size / (bbox1_size + bbox2_size - intersect_size);
        }
    }
    return overlap;
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

#ifdef DO_STAGE_0_CAFFE_OPT
KERNEL (detection_output_ref_stage_0_caffe_opt)(
    __global UNIT_TYPE* input_confidence,
    __global uchar *buffer0,
    __global int *buffer2)
{
    const int classId = get_global_id(0) * NUM_CLASSES_PER_ITEM;
    const int box_gid = get_global_id(1);
    const int batchId = get_global_id(2);

    int classes_leftover = ((NUM_CLASSES - (classId) >= NUM_CLASSES_PER_ITEM)) ?  0 : 1;
    int n_classes_this_item = classes_leftover ? (NUM_CLASSES - classId) : NUM_CLASSES_PER_ITEM;

    const int start_bid = box_gid * NUM_PRIORS_PER_ITEM;
    const int end_bid = min(start_bid + NUM_PRIORS_PER_ITEM, NUM_OF_PRIORS);

    __local char4 bit_mask[NUM_BIT_MASK];
    __local int4 block_num[NUM_PRIOR_BLOCKS];

    block_num[box_gid] = (int4)(0, 0, 0, 0);

    {
        int mask_id = start_bid / 8;
        for (int i = start_bid; i < end_bid; i += 8) {
            bit_mask[mask_id] = (char4)(0, 0, 0, 0);
            unroll_for (int bi = 0; bi < 8; bi++) {
                if ((i + bi) >= NUM_OF_PRIORS)
                    break;
                CMP_TYPE4 valid_scores = FUNC_CALL(filter_score4)(input_confidence, (i + bi), classId, batchId);
                bit_mask[mask_id] |= ((convert_char4(valid_scores)) << bi);
                block_num[box_gid] += convert_int4(valid_scores);
            }
            if (classes_leftover) {
                for (int c = n_classes_this_item; c < NUM_CLASSES_PER_ITEM; c++) {
                    bit_mask[mask_id][c] = 0;
                }
            }
            mask_id++;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        if (box_gid == 0 && get_local_id(1) == 0) {
            int4 acc_num = (int4)(0, 0, 0, 0);
            for (int i = 0; i < NUM_PRIOR_BLOCKS; i++) {
                int4 n = block_num[i];
                block_num[i] = acc_num;
                acc_num += n;
            }
            for (int c = 0; c < n_classes_this_item ; ++c) {
                buffer2[batchId * NUM_CLASSES_ACC + (classId + c)] = acc_num[c];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        int4 write_offsets = block_num[box_gid];
        int mask_id = start_bid >> 3;
        for (int i = start_bid; i < end_bid; i += 8) {
            for (int bi = 0; bi < 8; bi++) {
                char bitset = 1 << bi;
                if (all((bit_mask[mask_id] & bitset) == (char4)(0, 0, 0, 0)))
                    continue;
                UNIT_TYPE4 score4 = FUNC_CALL(get_score4)(input_confidence, (i + bi), classId, batchId);
                for (int c = 0; c < n_classes_this_item; c++) {
                    if ((bit_mask[mask_id][c] & bitset) == 0) continue;
                    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer0[(batchId * NUM_CLASSES + classId + c) * BUFFER_STRIDE];
                    SCORES_INFO score_info;
                    score_info.batchId = batchId;
                    score_info.classId = classId + c;
                    score_info.boxId = i + bi;
                    score_info.score = score4[c];
                    scoresList[write_offsets[c]] = score_info;
                    write_offsets[c]++;
                }
            }
            mask_id++;
        }
    }
}
#endif /* DO_STAGE_0_CAFFE_OPT */

#ifdef DO_STAGE_0_CAFFE
KERNEL (detection_output_ref_stage_0_caffe)(
    __global UNIT_TYPE* input_confidence,
    __global uchar *buffer0,
    __global int *buffer2)
{
    const int classId = get_global_id(0);
    const int box_gid = get_global_id(1);
    const int batchId = get_global_id(2);

    const int start_bid = box_gid * NUM_PRIORS_PER_ITEM;
    const int end_bid = min(start_bid + NUM_PRIORS_PER_ITEM, NUM_OF_PRIORS);

    __local char bit_mask[NUM_BIT_MASK];
    __local int block_num[NUM_PRIOR_BLOCKS];

    block_num[box_gid] = 0;

    {
        int mask_id = start_bid / 8;
        for (int i = start_bid; i < end_bid; i += 8) {
            bit_mask[mask_id] = 0;
            unroll_for (int bi = 0; bi < 8; bi++) {
                if ((i + bi) >= NUM_OF_PRIORS)
                    break;
                UNIT_TYPE score = FUNC_CALL(get_score)(input_confidence, (i + bi), classId, batchId);
                int valid = (score < 0) ? 0 : 1;
                bit_mask[mask_id] |= (valid << bi);
                block_num[box_gid] += valid;
            }
            mask_id++;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    {
        if (box_gid == 0 && get_local_id(1) == 0) {
            int acc_num = 0;
            for (int i = 0; i < NUM_PRIOR_BLOCKS; i++) {
                int n = block_num[i];
                block_num[i] = acc_num;
                acc_num += n;
            }
            buffer2[batchId * NUM_CLASSES_ACC + classId] = acc_num;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    {
        int write_offset = block_num[box_gid];
        int mask_id = start_bid >> 3;
        for (int i = start_bid; i < end_bid; i += 8) {
            for (int bi = 0; bi < 8; bi++) {
                char bitset = 1 << bi;
                if ((bit_mask[mask_id] & bitset) && ((i + bi) < NUM_OF_PRIORS)) {
                    UNIT_TYPE score = FUNC_CALL(get_score)(input_confidence, (i + bi), classId, batchId);
                    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer0[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];
                    SCORES_INFO score_info;
                    score_info.batchId = batchId;
                    score_info.classId = classId;
                    score_info.boxId = i + bi;
                    score_info.score = score;
                    scoresList[write_offset] = score_info;
                    write_offset++;
                }
            }
            mask_id++;
        }
    }
}
#endif /* DO_STAGE_0_CAFFE*/

#ifdef DO_STAGE_0_MXNET
KERNEL (detection_output_ref_stage_0_mxnet)(
    __global UNIT_TYPE* input_confidence,
    __global uchar *buffer0,
    volatile __global int *buffer2)
{
    const int batchId = get_global_id(0);
    const int priorId = get_global_id(1);

    const int scores_size_offset = batchId * NUM_OF_PRIORS + priorId;
    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer0[batchId * BUFFER_STRIDE];

    if (priorId == 0)
    {
        buffer2[batchId * NUM_CLASSES_ACC + NUM_CLASSES] = 0;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    int idx_max_score = FUNC_CALL(get_largest_score)(input_confidence, priorId, batchId);
    UNIT_TYPE score = FUNC_CALL(get_score)(input_confidence, priorId, idx_max_score, batchId);
    SCORES_INFO score_info;
    score_info.batchId = batchId;
    score_info.classId = idx_max_score;
    score_info.boxId = priorId;
    score_info.score = score;
    scoresList[priorId] = score_info;
    atomic_inc(&buffer2[batchId * NUM_CLASSES_ACC + NUM_CLASSES]);
}
#endif /* DO_STAGE_0_MXNET */

#ifdef DO_STAGE_1_CAFFE
KERNEL (detection_output_ref_stage_1_caffe)(
    __global uchar *buffer0,
    __global int *buffer2)
{
    const int batchId = get_global_id(0);
    const int classId = get_global_id(1);
    const int workItemId = get_global_id(2);
    const int localClassId = get_local_id(1);
    __local int __range[LOCAL_CLASS_NUM][LOCAL_WORK_NUM * 2];

    const int scoresInfoNum = buffer2[batchId * NUM_CLASSES_ACC + classId];

    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer0[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];

    if (workItemId == 0) {
        __range[localClassId][0] = 0;
        __range[localClassId][1] = (classId == BACKGROUND_LABEL_ID ? 0 : scoresInfoNum - 1);
    } else {
        __range[localClassId][workItemId * 2] = 0;
        __range[localClassId][workItemId * 2 + 1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int range_step = 2;
    const int first_id = workItemId * 2;
    for (int i = 0, maxWorkingNum = 1; i < PARTITION_STEP; ++i, maxWorkingNum *= 2, range_step *= 2) {
        if (workItemId < maxWorkingNum) {
            const int begin_id = __range[localClassId][first_id];
            const int end_id = __range[localClassId][first_id + 1];
            const int second_id = first_id + range_step;
            if (begin_id < end_id) {
                const int pivot = FUNC_CALL(partition)(scoresList, begin_id, end_id, true);
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
        FUNC_CALL(quickSortIterative)(scoresList, begin_id, end_id, true);
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    if (workItemId == 0) {
        if (TOP_K != -1 && TOP_K < scoresInfoNum) {
            buffer2[batchId * NUM_CLASSES_ACC + classId] = TOP_K;
        }
        if (classId == BACKGROUND_LABEL_ID) {
            buffer2[batchId * NUM_CLASSES_ACC + classId] = 0;
        }
    }
}
#endif /* DO_STAGE_1_CAFFE */

#ifdef DO_STAGE_1_MXNET
KERNEL (detection_output_ref_stage_1_mxnet)(
    __global uchar *buffer0,
    __global int *buffer2)
{
    const int batchId = get_global_id(0);
    const int workItemId = get_global_id(2);
    __local int __range[LOCAL_WORK_NUM * 2];

    const int scoresInfoNum = buffer2[batchId * NUM_CLASSES_ACC + NUM_CLASSES];
    if (scoresInfoNum < 2)
        return;

    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer0[batchId * BUFFER_STRIDE];

    if (workItemId == 0) {
        __range[0] = 0;
        __range[1] = scoresInfoNum - 1;
    } else {
        __range[workItemId * 2] = 0;
        __range[workItemId * 2 + 1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int range_step = 2;
    const int first_id = workItemId * 2;
    for (int i = 0; i < PARTITION_STEP; ++i, range_step *= 2) {
        if (workItemId <= i) {
            const int begin_id = __range[first_id];
            const int end_id = __range[first_id + 1];
            const int second_id = first_id + range_step;

            if (begin_id < end_id) {
                const int pivot = FUNC_CALL(partition)(scoresList, begin_id, end_id, true);
                __range[first_id     ] = begin_id;
                __range[first_id + 1 ] = max(pivot - 1, begin_id);
                __range[second_id    ] = min(pivot + 1, end_id);
                __range[second_id + 1] = end_id;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }

    const int begin_id = __range[first_id];
    const int end_id = __range[first_id + 1];
    if (begin_id < end_id) {
        FUNC_CALL(quickSortIterative)(scoresList, begin_id, end_id, true);
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    if (workItemId == 0 && (TOP_K != -1 && TOP_K < scoresInfoNum)) {
        buffer2[batchId * NUM_CLASSES_ACC + NUM_CLASSES] = TOP_K;
    }
}
#endif /* DO_STAGE_1_MXNET */

#ifdef DO_STAGE_2_CAFFE
KERNEL (detection_output_ref_stage_2_caffe)(
    __global UNIT_TYPE* input_location,
    __global UNIT_TYPE* input_prior_box,
    __global uchar *buffer0,
    __global uchar *buffer1,
    __global int *buffer2)
{
    const int batchId = get_global_id(0);
    const int classId = get_global_id(1);
    const int loc_label = ((SHARE_LOCATION)? 0 : classId);
    const int scoresInfoIdx = batchId * NUM_CLASSES_ACC + classId;

    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer0[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];
    __global SCORES_INFO *selectedScoresList = (__global SCORES_INFO*)&buffer1[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];

    const int scoresInfoNum = buffer2[scoresInfoIdx];

    int selectedBoxNum = 0;
    for (uint idx_score = 0; idx_score < scoresInfoNum; idx_score++)
    {
        bool keep = true;
        int idx = scoresList[idx_score].boxId;
        for (uint idx_indice = 0; idx_indice < selectedBoxNum; idx_indice++)
        {
            int kept_idx = selectedScoresList[idx_indice].boxId;
            UNIT_TYPE decoded_bbox1[4];
            FUNC_CALL(get_decoded_bbox)(decoded_bbox1, input_location, input_prior_box, idx, loc_label, batchId);
            UNIT_TYPE decoded_bbox2[4];
            FUNC_CALL(get_decoded_bbox)(decoded_bbox2, input_location, input_prior_box, kept_idx, loc_label, batchId);
            UNIT_TYPE overlap = FUNC_CALL(jaccardOverlap)(decoded_bbox1, decoded_bbox2);
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
            ++selectedBoxNum;
        }
    }
    buffer2[scoresInfoIdx] = selectedBoxNum;
}
#endif /* DO_STAGE_2_CAFFE */

#ifdef DO_STAGE_2_CAFFE_OPT
KERNEL (detection_output_ref_stage_2_caffe)(
    __global UNIT_TYPE* input_location,
    __global UNIT_TYPE* input_prior_box,
    __global uchar *buffer0,
    __global uchar *buffer1,
    __global int *buffer2)
{
    const int batchId = get_global_id(0);
    const int classId = get_global_id(1);
    const int loc_label = ((SHARE_LOCATION)? 0 : classId);
    const int scoresInfoIdx = batchId * NUM_CLASSES_ACC + classId;
    UNIT_TYPE decoded_bboxes[TOP_K * 4];

    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer0[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];
    __global SCORES_INFO *selectedScoresList = (__global SCORES_INFO*)&buffer1[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];

    const int scoresInfoNum = buffer2[scoresInfoIdx];

    int selectedBoxNum = 0;
    for (uint idx_score = 0; idx_score < scoresInfoNum; idx_score++)
    {
        bool keep = true;
        int idx = scoresList[idx_score].boxId;
        UNIT_TYPE decoded_bbox_cur[4];
        FUNC_CALL(get_decoded_bbox)(decoded_bbox_cur, input_location, input_prior_box, idx, loc_label, batchId);

        for (uint idx_indice = 0; idx_indice < selectedBoxNum; idx_indice++)
        {
            UNIT_TYPE decoded_bbox_kept[4] = { decoded_bboxes[4 * idx_indice],
                                           decoded_bboxes[4 * idx_indice + 1],
                                           decoded_bboxes[4 * idx_indice + 2],
                                           decoded_bboxes[4 * idx_indice + 3] };

            UNIT_TYPE overlap = FUNC_CALL(jaccardOverlap)(decoded_bbox_cur, decoded_bbox_kept);
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
            decoded_bboxes[4 * selectedBoxNum]     = decoded_bbox_cur[0];
            decoded_bboxes[4 * selectedBoxNum + 1] = decoded_bbox_cur[1];
            decoded_bboxes[4 * selectedBoxNum + 2] = decoded_bbox_cur[2];
            decoded_bboxes[4 * selectedBoxNum + 3] = decoded_bbox_cur[3];
            ++selectedBoxNum;
        }
    }

    buffer2[scoresInfoIdx] = selectedBoxNum;
}
#endif /* DO_STAGE_2_CAFFE_OPT */

#ifdef DO_STAGE_2_MXNET
KERNEL (detection_output_ref_stage_2_mxnet)(
    __global UNIT_TYPE* input_location,
    __global UNIT_TYPE* input_prior_box,
    __global uchar *buffer0,
    __global uchar *buffer1,
    __global int *buffer2)
{
    const int batchId = get_global_id(0);
    const int scoresInfoNum = buffer2[batchId * NUM_CLASSES_ACC + NUM_CLASSES];

    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer0[batchId * BUFFER_STRIDE];
    __global SCORES_INFO *selectedScoresList = (__global SCORES_INFO*)&buffer1[batchId * NUM_CLASSES * BUFFER_STRIDE];

    for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++)
    {
        buffer2[batchId * NUM_CLASSES_ACC + idx_class] = 0;
    }

    int selectedBoxNum = 0;
    for (uint idx_score = 0; idx_score < scoresInfoNum; idx_score++)
    {
        bool keep = true;
        int idx = scoresList[idx_score].boxId;
        int cls = scoresList[idx_score].classId;
        int loc_label = ((SHARE_LOCATION)? 0 : cls);
        int indice_offset = cls * NUM_OF_PRIORS;
        int scores_size_offset = batchId * NUM_CLASSES_ACC + cls;
        int cur_num_indice = buffer2[scores_size_offset];
        for (uint idx_indice = 0; idx_indice < cur_num_indice; idx_indice++)
        {
            int kept_idx = selectedScoresList[indice_offset + idx_indice].boxId;
            UNIT_TYPE decoded_bbox1[4];
            FUNC_CALL(get_decoded_bbox)(decoded_bbox1, input_location, input_prior_box, idx, loc_label, batchId);
            UNIT_TYPE decoded_bbox2[4];
            FUNC_CALL(get_decoded_bbox)(decoded_bbox2, input_location, input_prior_box, kept_idx, loc_label, batchId);
            UNIT_TYPE overlap = FUNC_CALL(jaccardOverlap)(decoded_bbox1, decoded_bbox2);
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
            selectedScoresList[indice_offset + cur_num_indice] = score_info;
            buffer2[scores_size_offset] = cur_num_indice + 1;
            ++selectedBoxNum;
        }
    }
    buffer2[batchId * NUM_CLASSES_ACC + NUM_CLASSES] = selectedBoxNum;
}
#endif /* DO_STAGE_2_MXNET */

#ifdef DO_STAGE_3_CAFFE
KERNEL (detection_output_ref_stage_final_caffe)(
    __global UNIT_TYPE* input_location,
    __global UNIT_TYPE* input_prior_box,
    __global UNIT_TYPE* output,
    __global uchar *buffer0,
    __global uchar *buffer1,
    __global int *buffer2)
{
    const int batchId = get_global_id(0);

    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer0[batchId * NUM_CLASSES * BUFFER_STRIDE];
    __global SCORES_INFO *selectedScoresList = (__global SCORES_INFO*)&buffer1[0];

    const int total_det = FUNC_CALL(get_accumulated_detections)(buffer2, batchId);
    buffer2[batchId * NUM_CLASSES_ACC + NUM_CLASSES] = total_det;
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (KEEP_TOP_K > -1 && total_det > KEEP_TOP_K)
    {
        int num_det = 0;
        for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++)
        {
            int scores_size_offset = batchId * NUM_CLASSES_ACC + idx_class;
            const int acc_num = buffer2[scores_size_offset];
            int scores_offset = (batchId * NUM_CLASSES * NUM_OF_PRIORS) + idx_class * NUM_OF_PRIORS;

            for (uint idx_score = 0; idx_score < acc_num; idx_score++)
            {
                SCORES_INFO score_info;
                score_info = selectedScoresList[scores_offset + idx_score];
                scoresList[num_det + idx_score] = score_info;
            }
            num_det += acc_num;
            buffer2[scores_size_offset] = 0;
        }

        FUNC_CALL(quickSortIterative)(scoresList, 0, num_det - 1, true);

        for (uint idx_num_det = 0; idx_num_det < KEEP_TOP_K; idx_num_det++)
        {
            SCORES_INFO score_info;
            score_info = scoresList[idx_num_det];
            int scores_size_offset = batchId * NUM_CLASSES_ACC + score_info.classId;
            int acc_num = buffer2[scores_size_offset];
            int scores_offset = (batchId * NUM_CLASSES * NUM_OF_PRIORS) + score_info.classId * NUM_OF_PRIORS + acc_num;
            selectedScoresList[scores_offset] = score_info;
            buffer2[scores_size_offset] = (acc_num + 1);
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    const int startIdx = FUNC_CALL(get_start_idx)(buffer2, batchId);
    int outputIdx = 0;
    for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++)
    {
        int scores_size_offset = batchId * NUM_CLASSES_ACC + idx_class;
        const int acc_num = buffer2[scores_size_offset];
        int scores_offset = (batchId * NUM_CLASSES * NUM_OF_PRIORS) + idx_class * NUM_OF_PRIORS;
        for (uint idx_score = 0; idx_score < acc_num; idx_score++)
        {
            SCORES_INFO score_info;
            score_info = selectedScoresList[scores_offset + idx_score];
            const int idx = startIdx + outputIdx;

            output[idx * OUTPUT_ROW_SIZE] = TO_UNIT_TYPE(score_info.batchId);
            output[idx * OUTPUT_ROW_SIZE + 1] = TO_UNIT_TYPE((DECREASE_LABEL_ID) ? score_info.classId - 1 : score_info.classId);
            output[idx * OUTPUT_ROW_SIZE + 2] = TO_UNIT_TYPE(score_info.score);
            UNIT_TYPE decoded_bbox[4];
            const uint loc_label = ((SHARE_LOCATION)? 0 : score_info.classId);
            FUNC_CALL(get_decoded_bbox)(decoded_bbox, input_location, input_prior_box, score_info.boxId, loc_label, score_info.batchId);
            UNIT_TYPE xmin = decoded_bbox[0];
            UNIT_TYPE ymin = decoded_bbox[1];
            UNIT_TYPE xmax = decoded_bbox[2];
            UNIT_TYPE ymax = decoded_bbox[3];
            if (CLIP_AFTER_NMS) {
                xmin = max(TO_UNIT_TYPE(0.0), min(TO_UNIT_TYPE(1.0), xmin));
                ymin = max(TO_UNIT_TYPE(0.0), min(TO_UNIT_TYPE(1.0), ymin));
                xmax = max(TO_UNIT_TYPE(0.0), min(TO_UNIT_TYPE(1.0), xmax));
                ymax = max(TO_UNIT_TYPE(0.0), min(TO_UNIT_TYPE(1.0), ymax));
            }
            vstore4((UNIT_TYPE4)(xmin, ymin, xmax, ymax), 0, output + idx * OUTPUT_ROW_SIZE + 3);
            outputIdx++;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    if(batchId == 0)
    {
        const int final_detections = FUNC_CALL(get_final_detections)(buffer2);
        unroll_for (uint i = final_detections; i < NUM_OF_IMAGES * KEEP_TOP_K; i++)
        {
            output[i * OUTPUT_ROW_SIZE] = (i == final_detections ? -1.0 : 0.0);
            vstore4((UNIT_TYPE4)(0.0, 0.0, 0.0, 0.0), 0, output + i * OUTPUT_ROW_SIZE + 1);
            vstore2((UNIT_TYPE2)(0.0, 0.0), 0, output + i * OUTPUT_ROW_SIZE + 5);
        }
    }
}
#endif  /* DO_STAGE_3_CAFFE */

#ifdef DO_STAGE_3_MXNET
KERNEL (detection_output_ref_stage_final_mxnet)(
    __global UNIT_TYPE* input_location,
    __global UNIT_TYPE* input_prior_box,
    __global UNIT_TYPE* output,
    __global uchar *buffer0,
    __global uchar *buffer1,
    __global int *buffer2)
{
    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer0[0];
    __global SCORES_INFO *selectedScoresList = (__global SCORES_INFO*)&buffer1[0];

    for (uint idx_image = 0; idx_image < NUM_OF_IMAGES; idx_image++)
    {
        const int total_det = buffer2[idx_image * NUM_CLASSES_ACC + NUM_CLASSES];

        if (KEEP_TOP_K > -1 && total_det > KEEP_TOP_K)
        {
            int num_det = 0;
            for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++)
            {
                int scores_size_offset = idx_image * NUM_CLASSES_ACC + idx_class;
                const int acc_num = buffer2[scores_size_offset];
                int selected_scores_offset = (idx_image * NUM_CLASSES * NUM_OF_PRIORS) + idx_class * NUM_OF_PRIORS;
                int scores_offset = idx_image * NUM_OF_PRIORS + num_det;

                for (uint idx_score = 0; idx_score < acc_num; idx_score++)
                {
                    SCORES_INFO score_info;
                    score_info = selectedScoresList[selected_scores_offset + idx_score];
                    scoresList[scores_offset + idx_score] = score_info;
                }
                num_det += acc_num;
                buffer2[scores_size_offset] = 0;
            }
            FUNC_CALL(quickSortIterative)(&scoresList[idx_image * NUM_OF_PRIORS], 0, num_det - 1, true);

            for (uint idx_num_det = 0; idx_num_det < KEEP_TOP_K; idx_num_det++)
            {
                SCORES_INFO score_info;
                score_info = scoresList[idx_image * NUM_OF_PRIORS + idx_num_det];
                int scores_size_offset = idx_image * NUM_CLASSES_ACC + score_info.classId;
                int acc_num = buffer2[scores_size_offset];
                int scores_offset = (idx_image * NUM_CLASSES * NUM_OF_PRIORS) + score_info.classId * NUM_OF_PRIORS + acc_num;
                selectedScoresList[scores_offset] = score_info;
                buffer2[scores_size_offset] = (acc_num + 1);
            }
        }
    }

    int count = 0;
    for (uint idx_image = 0; idx_image < NUM_OF_IMAGES; idx_image++)
    {
        for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++)
        {
            int scores_size_offset = idx_image * NUM_CLASSES_ACC + idx_class;
            int acc_num = buffer2[scores_size_offset];
            int scores_offset = (idx_image * NUM_CLASSES * NUM_OF_PRIORS) + idx_class * NUM_OF_PRIORS;
            int loc_label = ((SHARE_LOCATION)? 0 : idx_class);
            for (uint idx_score = 0; idx_score < acc_num; idx_score++)
            {
                SCORES_INFO score_info;
                score_info = selectedScoresList[scores_offset + idx_score];
                output[count * OUTPUT_ROW_SIZE] = TO_UNIT_TYPE(score_info.batchId);
                output[count * OUTPUT_ROW_SIZE + 1] = TO_UNIT_TYPE((DECREASE_LABEL_ID) ? score_info.classId - 1 : score_info.classId);
                output[count * OUTPUT_ROW_SIZE + 2] = TO_UNIT_TYPE(score_info.score);
                UNIT_TYPE decoded_bbox[4];
                FUNC_CALL(get_decoded_bbox)(decoded_bbox, input_location, input_prior_box, score_info.boxId, loc_label, idx_image);
                UNIT_TYPE xmin = decoded_bbox[0];
                UNIT_TYPE ymin = decoded_bbox[1];
                UNIT_TYPE xmax = decoded_bbox[2];
                UNIT_TYPE ymax = decoded_bbox[3];

                if (CLIP_AFTER_NMS) {
                    xmin = max(TO_UNIT_TYPE(0.0), min(TO_UNIT_TYPE(1.0), xmin));
                    ymin = max(TO_UNIT_TYPE(0.0), min(TO_UNIT_TYPE(1.0), ymin));
                    xmax = max(TO_UNIT_TYPE(0.0), min(TO_UNIT_TYPE(1.0), xmax));
                    ymax = max(TO_UNIT_TYPE(0.0), min(TO_UNIT_TYPE(1.0), ymax));
                }
                vstore4((UNIT_TYPE4)(xmin, ymin, xmax, ymax), 0, output + count * OUTPUT_ROW_SIZE + 3);
                ++count;
            }
        }
    }

    if (count < NUM_OF_IMAGES * KEEP_TOP_K)
    {
        output[count * OUTPUT_ROW_SIZE] = -1.0;
    }
}
#endif  /* DO_STAGE_3_MXNET */
