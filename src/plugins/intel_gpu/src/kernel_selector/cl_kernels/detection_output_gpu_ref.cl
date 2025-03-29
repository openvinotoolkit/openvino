// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/detection_output_common.cl"

// DetectionOuput - performs non-maximuim suppression to generate the detection output
//                  using information on location and confidence predictions.
//
// Below is a set of 4 kernels:
//   - detection_output_stage_0_scores_(caffe/mxnet)
//   - detection_output_stage_1_sort_(caffe/mxnet)
//   - detection_output_stage_2_nms_(caffe/mxnet)
//   - detection_output_stage_final_(caffe/mxnet)
// that can perform detection output operation in two modes determined by decrease_label_id.
//
// Caffe-style NMS mode:
//   In this mode _caffe kernels are used.
//
//   detection_output_stage_0_scores_caffe should be first enqueued, provided extra global memory
//   on second and third input.
//     This kernel will calculate detections whose confidences are larger than a threshold and
//     the number of detections per each class for each work-groups and store it into global memory.
//
//   detection_output_stage_1_sort_caffe should be next enqueued in order to sort detections.
//     This kernel expects on first and second input global memory from the result of previous kernel.
//     In this kernel, detections for each class are sorted using iterative quick sort from each
//     work-groups and store it into global memory. If the total of detections per each batch is
//     greater than TOP_K, it is stored as TOP_K into global memory.
//
//   detection_output_stage_2_nms_caffe sould be next enqueued with provided buffers with outputs
//   from previous kernel and inputs(location and prior_box).
//     This kernel will select detections per each class using non-maximum suppresion for work-goups
//     and store it into global memory. During NMS, box coordinates of detections are calculated
//     from inputs(location and prior_box) using bounding box decoding.
//
//   Finally detection_output_stage_final_caffe should be enqueued with provided buffers with outputs
//   from previous kernel and output using single work-group.
//     This kernel will produce the results of the final detections form the result of previous kernel.
//     If the total of detections per each batch is greater than KEEP_TOP_K, detections are sorted using
//     iterative quick sort and it is stored as KEEP_TOP_K. Final detections contain information about
//     filtered detection described with 7 elements [batch_id, class_id, confidence, x_1, y_1, x_2, y_2].
//
// =================================================================================================================
// Required jit constants:
// -----------------------------------------------------------------------------------------------------------------
// BUFFER_STRIDE         - buffer size per class
// NUM_BIT_MASK          - bit mask size that can be processed per work-group
// NUM_PRIORS_PER_ITEM   - number of prior boxes that can be processed per work-item
// NUM_PRIOR_BLOCKS      - local memory size that can handle the number of detections accumulated per work-group
// LOCAL_CLASS_NUM       - number of class that can be process per work-item
// LOCAL_WORK_NUM        - number of work-items that can be processed simultaneously
// PARTITION_STEP        - loop size that will perform partition to calculalte pivot and store it into local memory
// LOCAL_BATCHES_NUM     - number of batch that can be process per work-group
// =================================================================================================================

#define NUM_CLASSES_ACC (NUM_CLASSES + 2)

typedef struct __attribute__((__packed__)) {
    short classId;
    int boxId;
    INPUT1_TYPE score;
} FUNC(Scores);

#define SCORES_INFO FUNC(Scores)

inline void FUNC(swap_scores_info)(__global SCORES_INFO* a, __global SCORES_INFO* b) {
    SCORES_INFO temp = *a;
    *a = *b;
    *b = temp;
}

inline int FUNC(partition)(__global SCORES_INFO* arr, int l, int h, bool use_custom_comp) {
    INPUT1_TYPE pivotScore = arr[h].score;
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

inline void FUNC(bubbleSortIterative)(__global SCORES_INFO* arr, int l, int h) {
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

inline void FUNC(quickSortIterative)(__global SCORES_INFO* arr,
                                     int l, int h,
#ifdef USE_LOCAL_MEMORY_FOR_STACK
                                     __local int* stack,
#endif
                                     bool use_custom_comp) {
#ifndef USE_LOCAL_MEMORY_FOR_STACK
    // Create an auxiliary stack
    int stack[QUICK_SORT_STACK_SIZE];
#endif
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
            if (top >= (QUICK_SORT_STACK_SIZE - 1)) {
                FUNC_CALL(bubbleSortIterative)(arr, l, p - 1);
            } else {
                stack[++top] = l;
                stack[++top] = p - 1;
            }
        }

        // If there are elements on right side of pivot,
        // then push right side to stack
        if (p + 1 < h) {
            if (top >= (QUICK_SORT_STACK_SIZE - 1)) {
                FUNC_CALL(bubbleSortIterative)(arr, p + 1, h);
            } else {
                stack[++top] = p + 1;
                stack[++top] = h;
            }
        }
    }
}

inline int FUNC(get_accumulated_detections)(__global int* size_buf, int batch_id) {
    int acc_num = 0;
    for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++)
    {
        acc_num += size_buf[batch_id * NUM_CLASSES_ACC + idx_class];
    }
    return acc_num;
}

inline int FUNC(get_start_idx)(__global int* size_buf, int batch_id, int offset) {
    int start_idx = 0;
    for (uint idx_batch = 0; idx_batch < batch_id; idx_batch++)
    {
        const int num_det = size_buf[idx_batch * NUM_CLASSES_ACC + NUM_CLASSES + offset];
        start_idx += (num_det > KEEP_TOP_K ? KEEP_TOP_K: num_det);
    }
    return start_idx;
}

inline int FUNC(get_final_detections)(__global int* size_buf) {
    int final_detections = 0;
    for (uint idx_image = 0; idx_image < NUM_OF_IMAGES; idx_image++)
    {
        const int num_det = size_buf[idx_image * NUM_CLASSES_ACC + NUM_CLASSES];
        final_detections += (num_det > KEEP_TOP_K ? KEEP_TOP_K: num_det);
    }
    return final_detections;
}

inline INPUT0_TYPE FUNC(jaccardOverlap)(INPUT0_TYPE* bbox1, INPUT0_TYPE* bbox2) {
    INPUT0_TYPE overlap = 0.0;
    bool intersecting = (bbox1[0] < bbox2[2]) & (bbox2[0] < bbox1[2]) & (bbox1[1] < bbox2[3]) & (bbox2[1] < bbox1[3]);

    if (intersecting)
    {
        const INPUT0_TYPE intersect_width = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]);
        const INPUT0_TYPE intersect_height = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]);
        if (intersect_width > 0 && intersect_height > 0) {
            const INPUT0_TYPE intersect_size = intersect_width * intersect_height;
            const INPUT0_TYPE bbox1_size = (bbox1[2] - bbox1[0]) * (bbox1[3]- bbox1[1]);
            const INPUT0_TYPE bbox2_size = (bbox2[2] - bbox2[0]) * (bbox2[3]- bbox2[1]);
            overlap = intersect_size / (bbox1_size + bbox2_size - intersect_size);
        }
    }
    return overlap;
}

inline uint FUNC(get_confidence_offset)(const uint idx_prior, const uint idx_class, const uint idx_image) {
    return (idx_prior * NUM_CLASSES + idx_image * NUM_OF_PRIORS * NUM_CLASSES + idx_class) * CONF_XY_SIZE_PRODUCT + CONF_PADDING;
}

inline uint FUNC(get_largest_score)(__global INPUT1_TYPE* input_confidence, const uint idx_prior, const uint idx_image) {
    const uint idx_start = (BACKGROUND_LABEL_ID == 0 ? 1 : 0);
    uint offset = FUNC_CALL(get_confidence_offset)(idx_prior, idx_start, idx_image);
    INPUT1_TYPE max_score = input_confidence[offset];
    uint idx = idx_start;

    for (uint j = idx_start; j < NUM_CLASSES; j++)
    {
        offset = FUNC_CALL(get_confidence_offset)(idx_prior, j, idx_image);
        INPUT1_TYPE score = input_confidence[offset];
        if (score > max_score) {
            max_score = score;
            idx = j;
        }
    }
    return idx;
}

#ifdef DO_STAGE_0_CAFFE_OPT
KERNEL (detection_output_stage_0_scores_caffe)(__global INPUT1_TYPE* input_confidence,
                                               __global uchar *buffer0,
                                               __global int *buffer1) {
    const int classId = (int)get_global_id(0) * NUM_CLASSES_PER_ITEM;
    const int box_gid = get_global_id(1);
    const int batchId = get_global_id(2);

    int classes_leftover = ((NUM_CLASSES - (classId) >= NUM_CLASSES_PER_ITEM)) ?  0 : 1;
    int n_classes_this_item = classes_leftover ? (NUM_CLASSES - classId) : NUM_CLASSES_PER_ITEM;

    const int start_bid = box_gid * NUM_PRIORS_PER_ITEM;
    const int end_bid = min(start_bid + NUM_PRIORS_PER_ITEM, NUM_OF_PRIORS);

    __local char4 bit_mask[NUM_BIT_MASK];
    __local int4 block_num[NUM_PRIOR_BLOCKS];

    {
        // to prevent access array out of range
        if (start_bid < end_bid)
            block_num[box_gid] = (int4)(0, 0, 0, 0);
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
                buffer1[batchId * NUM_CLASSES_ACC + (classId + c)] = acc_num[c];
            }
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    {
        int4 write_offsets = (int4)(0, 0, 0, 0);
        if (start_bid < end_bid)
            write_offsets = block_num[box_gid];
        int mask_id = start_bid >> 3;
        for (int i = start_bid; i < end_bid; i += 8) {
            for (int bi = 0; bi < 8; bi++) {
                char bitset = 1 << bi;
                if (all((bit_mask[mask_id] & bitset) == (char4)(0, 0, 0, 0)))
                    continue;
                INPUT_TYPE4 score4 = FUNC_CALL(get_score4)(input_confidence, (i + bi), classId, batchId);
                for (int c = 0; c < n_classes_this_item; c++) {
                    if ((bit_mask[mask_id][c] & bitset) == 0) continue;
                    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer0[(batchId * NUM_CLASSES + classId + c) * BUFFER_STRIDE];
                    SCORES_INFO score_info;
                    score_info.classId = (short)(classId + c);
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
KERNEL (detection_output_stage_0_scores_caffe)(__global INPUT1_TYPE* input_confidence,
                                               __global uchar *buffer0,
                                               __global int *buffer1) {
    const int classId = get_global_id(0);
    const int box_gid = get_global_id(1);
    const int batchId = get_global_id(2);

    const int start_bid = box_gid * NUM_PRIORS_PER_ITEM;
    const int end_bid = min(start_bid + NUM_PRIORS_PER_ITEM, NUM_OF_PRIORS);

    __local char bit_mask[NUM_BIT_MASK];
    __local int block_num[NUM_PRIOR_BLOCKS];

    {
        // to prevent access array out of range
        if (start_bid < end_bid)
            block_num[box_gid] = 0;
        int mask_id = start_bid / 8;
        for (int i = start_bid; i < end_bid; i += 8) {
            bit_mask[mask_id] = 0;
            unroll_for (int bi = 0; bi < 8; bi++) {
                if ((i + bi) >= NUM_OF_PRIORS)
                    break;
                INPUT1_TYPE score = FUNC_CALL(get_score)(input_confidence, (i + bi), classId, batchId);
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
            buffer1[batchId * NUM_CLASSES_ACC + classId] = acc_num;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    {
        int write_offset = 0;
        if (start_bid < end_bid)
            write_offset = block_num[box_gid];
        int mask_id = start_bid >> 3;
        for (int i = start_bid; i < end_bid; i += 8) {
            for (int bi = 0; bi < 8; bi++) {
                char bitset = 1 << bi;
                if ((bit_mask[mask_id] & bitset) && ((i + bi) < NUM_OF_PRIORS)) {
                    INPUT1_TYPE score = FUNC_CALL(get_score)(input_confidence, (i + bi), classId, batchId);
                    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer0[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];
                    SCORES_INFO score_info;
                    score_info.classId = (short)classId;
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
KERNEL (detection_output_stage_0_scores_mxnet)(__global INPUT1_TYPE* input_confidence,
                                               __global uchar *buffer0,
                                               volatile __global int *buffer2) {
    const int batchId = get_global_id(0);
    const int priorId = get_global_id(1);

    const int scores_size_offset = batchId * NUM_OF_PRIORS + priorId;
    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer0[batchId * BUFFER_STRIDE];

    if (priorId == 0) {
        buffer2[batchId * NUM_CLASSES_ACC + NUM_CLASSES] = 0;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    int idx_max_score = FUNC_CALL(get_largest_score)(input_confidence, priorId, batchId);
    INPUT1_TYPE score = FUNC_CALL(get_score)(input_confidence, priorId, idx_max_score, batchId);
    SCORES_INFO score_info;
    score_info.classId = (short)idx_max_score;
    score_info.boxId = priorId;
    score_info.score = score;
    scoresList[priorId] = score_info;
    atomic_inc(&buffer2[batchId * NUM_CLASSES_ACC + NUM_CLASSES]);
}
#endif /* DO_STAGE_0_MXNET */

#ifdef DO_STAGE_1_CAFFE
KERNEL (detection_output_stage_1_sort_caffe)(__global uchar *buffer0,
                                             __global int *buffer1) {
    const int batchId = get_global_id(0);
    const int classId = get_global_id(1);
    const int workItemId = get_global_id(2);
    const int localClassId = get_local_id(1);
    __local int __range[LOCAL_CLASS_NUM][LOCAL_WORK_NUM * 2];

#ifdef USE_LOCAL_MEMORY_FOR_STACK
    // Create an auxiliary stack for QuickSort
    __local int stack[QUICK_SORT_STACK_SIZE * LOCAL_CLASS_NUM * LOCAL_WORK_NUM];
    __local int *stack_pointer = stack + workItemId * QUICK_SORT_STACK_SIZE + localClassId * LOCAL_WORK_NUM * QUICK_SORT_STACK_SIZE;
#endif

    const int scoresInfoNum = buffer1[batchId * NUM_CLASSES_ACC + classId];

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
#ifdef USE_LOCAL_MEMORY_FOR_STACK
        FUNC_CALL(quickSortIterative)(scoresList, begin_id, end_id, stack_pointer, true);
#else
        FUNC_CALL(quickSortIterative)(scoresList, begin_id, end_id, true);
#endif
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    if (workItemId == 0) {
        if (TOP_K != -1 && TOP_K < scoresInfoNum) {
            buffer1[batchId * NUM_CLASSES_ACC + classId] = TOP_K;
        }
        if (classId == BACKGROUND_LABEL_ID) {
            buffer1[batchId * NUM_CLASSES_ACC + classId] = 0;
        }
    }
}
#endif /* DO_STAGE_1_CAFFE */

#ifdef DO_STAGE_1_MXNET
KERNEL (detection_output_stage_1_sort_mxnet)(__global uchar *buffer0,
                                             __global int *buffer2) {
    const int batchId = get_global_id(0);
    const int workItemId = get_global_id(2);
    __local int __range[LOCAL_WORK_NUM * 2];

    // Create an auxiliary stack for QuickSort
    __local int stack[QUICK_SORT_STACK_SIZE];

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
        FUNC_CALL(quickSortIterative)(scoresList, begin_id, end_id, stack, true);
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    if (workItemId == 0 && (TOP_K != -1 && TOP_K < scoresInfoNum)) {
        buffer2[batchId * NUM_CLASSES_ACC + NUM_CLASSES] = TOP_K;
    }
}
#endif /* DO_STAGE_1_MXNET */

#ifdef DO_STAGE_2_CAFFE
KERNEL (detection_output_stage_2_nms_caffe)(__global INPUT0_TYPE* input_location,
                                            __global INPUT2_TYPE* input_prior_box,
                                            __global uchar *buffer0,
                                            __global int *buffer1) {
    const int batchId = get_global_id(0);
    const int classId = get_global_id(1);
    const int loc_label = ((SHARE_LOCATION)? 0 : classId);
    const int scoresInfoIdx = batchId * NUM_CLASSES_ACC + classId;

    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer0[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];

    const int scoresInfoNum = buffer1[scoresInfoIdx];

    int selectedBoxNum = 0;
    for (uint idx_score = 0; idx_score < scoresInfoNum; idx_score++) {
        bool keep = true;
        int idx = scoresList[idx_score].boxId;
        for (uint idx_indice = 0; idx_indice < selectedBoxNum; idx_indice++) {
            int kept_idx = scoresList[idx_indice].boxId;
            INPUT0_TYPE decoded_bbox1[4];
            FUNC_CALL(get_decoded_bbox)(decoded_bbox1, input_location, input_prior_box, idx, loc_label, batchId);
            INPUT0_TYPE decoded_bbox2[4];
            FUNC_CALL(get_decoded_bbox)(decoded_bbox2, input_location, input_prior_box, kept_idx, loc_label, batchId);
            INPUT0_TYPE overlap = FUNC_CALL(jaccardOverlap)(decoded_bbox1, decoded_bbox2);
            if (overlap > NMS_THRESHOLD) {
                keep = false;
                break;
            }
        }
        if (keep) {
            scoresList[selectedBoxNum] = scoresList[idx_score];
            ++selectedBoxNum;
        }
    }
    buffer1[scoresInfoIdx] = selectedBoxNum;
}
#endif /* DO_STAGE_2_CAFFE */

#ifdef DO_STAGE_2_CAFFE_OPT
KERNEL (detection_output_stage_2_nms_caffe)(__global INPUT0_TYPE* input_location,
                                            __global INPUT2_TYPE* input_prior_box,
                                            __global uchar *buffer0,
                                            __global int *buffer1) {
    const int batchId = get_global_id(0);
    const int classId = get_global_id(1);
    const int loc_label = ((SHARE_LOCATION)? 0 : classId);
    const int scoresInfoIdx = batchId * NUM_CLASSES_ACC + classId;
#ifdef USE_LOCAL_MEMORY
    __local   INPUT0_TYPE decoded_bboxes[TOP_K * 4];
#else
    __private INPUT0_TYPE decoded_bboxes[TOP_K * 4];
#endif

    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer0[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];

    const int scoresInfoNum = buffer1[scoresInfoIdx];

    int selectedBoxNum = 0;
    for (uint idx_score = 0; idx_score < scoresInfoNum; idx_score++) {
        bool keep = true;
        int idx = scoresList[idx_score].boxId;
        INPUT0_TYPE decoded_bbox_cur[4];
        FUNC_CALL(get_decoded_bbox)(decoded_bbox_cur, input_location, input_prior_box, idx, loc_label, batchId);

        for (uint idx_indice = 0; idx_indice < selectedBoxNum; idx_indice++) {
            INPUT0_TYPE decoded_bbox_kept[4] = { decoded_bboxes[4 * idx_indice],
                                                 decoded_bboxes[4 * idx_indice + 1],
                                                 decoded_bboxes[4 * idx_indice + 2],
                                                 decoded_bboxes[4 * idx_indice + 3] };

            INPUT0_TYPE overlap = FUNC_CALL(jaccardOverlap)(decoded_bbox_cur, decoded_bbox_kept);
            if (overlap > NMS_THRESHOLD) {
                keep = false;
                break;
            }
        }
        if (keep) {
            scoresList[selectedBoxNum] = scoresList[idx_score];
            decoded_bboxes[4 * selectedBoxNum]     = decoded_bbox_cur[0];
            decoded_bboxes[4 * selectedBoxNum + 1] = decoded_bbox_cur[1];
            decoded_bboxes[4 * selectedBoxNum + 2] = decoded_bbox_cur[2];
            decoded_bboxes[4 * selectedBoxNum + 3] = decoded_bbox_cur[3];
            ++selectedBoxNum;
        }
    }
    buffer1[scoresInfoIdx] = selectedBoxNum;
}
#endif /* DO_STAGE_2_CAFFE_OPT */

#ifdef DO_STAGE_2_MXNET
KERNEL (detection_output_stage_2_nms_mxnet)(__global INPUT0_TYPE* input_location,
                                            __global INPUT2_TYPE* input_prior_box,
                                            __global uchar *buffer0,
                                            __global uchar *buffer1,
                                            __global int *buffer2) {
    const int batchId = get_global_id(0);
    const int scoresInfoNum = buffer2[batchId * NUM_CLASSES_ACC + NUM_CLASSES];

    __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer0[batchId * BUFFER_STRIDE];
    __global SCORES_INFO *selectedScoresList = (__global SCORES_INFO*)&buffer1[batchId * NUM_CLASSES * BUFFER_STRIDE];

    for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++) {
        buffer2[batchId * NUM_CLASSES_ACC + idx_class] = 0;
    }

    int selectedBoxNum = 0;
    for (uint idx_score = 0; idx_score < scoresInfoNum; idx_score++) {
        bool keep = true;
        int idx = scoresList[idx_score].boxId;
        int cls = (int)scoresList[idx_score].classId;
        int loc_label = ((SHARE_LOCATION)? 0 : cls);
        int indice_offset = cls * NUM_OF_PRIORS;
        int scores_size_offset = batchId * NUM_CLASSES_ACC + cls;
        int cur_num_indice = buffer2[scores_size_offset];
        for (uint idx_indice = 0; idx_indice < cur_num_indice; idx_indice++) {
            int kept_idx = selectedScoresList[indice_offset + idx_indice].boxId;
            INPUT0_TYPE decoded_bbox1[4];
            FUNC_CALL(get_decoded_bbox)(decoded_bbox1, input_location, input_prior_box, idx, loc_label, batchId);
            INPUT0_TYPE decoded_bbox2[4];
            FUNC_CALL(get_decoded_bbox)(decoded_bbox2, input_location, input_prior_box, kept_idx, loc_label, batchId);
            INPUT0_TYPE overlap = FUNC_CALL(jaccardOverlap)(decoded_bbox1, decoded_bbox2);
            if (overlap > NMS_THRESHOLD) {
                keep = false;
                break;
            }
        }
        if (keep) {
            SCORES_INFO score_info;
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
KERNEL (detection_output_stage_final_caffe)(__global INPUT0_TYPE* input_location,
                                            __global INPUT2_TYPE* input_prior_box,
                                            __global OUTPUT_TYPE* output,
                                            __global uchar *buffer0,
                                            __global int *buffer1) {
    const int batchId = get_global_id(0);

#ifdef USE_LOCAL_MEMORY_FOR_STACK
    // Create an auxiliary stack for QuickSort
    __local int stack[QUICK_SORT_STACK_SIZE];
    __local int *stack_pointer = stack + batchId * QUICK_SORT_STACK_SIZE;
#endif

    const int total_det = FUNC_CALL(get_accumulated_detections)(buffer1, batchId);
    buffer1[batchId * NUM_CLASSES_ACC + NUM_CLASSES] = total_det;
    // the total number of detections is also stored in the extra space of buffer
    // for case where the number of detections is larger than keep_top_k
    buffer1[batchId * NUM_CLASSES_ACC + NUM_CLASSES + 1] = total_det;

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (KEEP_TOP_K > -1 && total_det > KEEP_TOP_K) {
        __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer0[0];
        int num_det = 0;
        int scores_offset = (batchId * NUM_CLASSES * NUM_OF_PRIORS);
        int scores_size_offset = batchId * NUM_CLASSES_ACC;

        for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++) {
            const int acc_num = buffer1[scores_size_offset + idx_class];

            for (uint idx_score = 0; idx_score < acc_num; idx_score++) {
                SCORES_INFO score_info;
                score_info = *((__global SCORES_INFO*)&buffer0[(batchId * NUM_CLASSES + idx_class) * BUFFER_STRIDE] + idx_score);
                scoresList[scores_offset + num_det + idx_score] = score_info;
            }
            num_det += acc_num;
            buffer1[scores_size_offset + idx_class] = 0;
        }

#ifdef USE_LOCAL_MEMORY_FOR_STACK
        FUNC_CALL(quickSortIterative)(scoresList + scores_offset, 0, num_det - 1, stack_pointer, true);
#else
        FUNC_CALL(quickSortIterative)(scoresList + scores_offset, 0, num_det - 1, true);
#endif

        // recalculate valid items for each class
        for (uint idx_num_det = 0; idx_num_det < KEEP_TOP_K; idx_num_det++) {
            SCORES_INFO score_info = scoresList[scores_offset + idx_num_det];
            buffer1[scores_size_offset + score_info.classId]++;
        }

        // calculate starting point of each class
        // store the current number of detections for buffer reuse
        int prev_offset = buffer1[scores_size_offset];
        buffer1[scores_size_offset] = 0;
        for (int i = 1; i < NUM_CLASSES_ACC; ++i) {
            int cur_offset = buffer1[scores_size_offset + i];
            buffer1[scores_size_offset + i] = buffer1[scores_size_offset + i - 1] + prev_offset;
            prev_offset = cur_offset;
        }

        barrier(CLK_GLOBAL_MEM_FENCE);

        const int startIdx = FUNC_CALL(get_start_idx)(buffer1, batchId, 1);
        for (uint idx_num_det = 0; idx_num_det < KEEP_TOP_K; idx_num_det++) {
            SCORES_INFO score_info;
            score_info = scoresList[scores_offset + idx_num_det];
            const int idx = startIdx + buffer1[scores_size_offset + score_info.classId];
            output[idx * OUTPUT_ROW_SIZE] = TO_OUTPUT_TYPE(batchId);
            output[idx * OUTPUT_ROW_SIZE + 1] = TO_OUTPUT_TYPE((DECREASE_LABEL_ID) ? score_info.classId - 1 : score_info.classId);
            output[idx * OUTPUT_ROW_SIZE + 2] = TO_OUTPUT_TYPE(score_info.score);

            INPUT0_TYPE decoded_bbox[4];
            const uint loc_label = ((SHARE_LOCATION)? 0 : score_info.classId);
            FUNC_CALL(get_decoded_bbox)(decoded_bbox, input_location, input_prior_box, score_info.boxId, loc_label, batchId);
            INPUT0_TYPE xmin = decoded_bbox[0];
            INPUT0_TYPE ymin = decoded_bbox[1];
            INPUT0_TYPE xmax = decoded_bbox[2];
            INPUT0_TYPE ymax = decoded_bbox[3];
            if (CLIP_AFTER_NMS) {
                xmin = max(TO_INPUT0_TYPE(0.0), min(TO_INPUT0_TYPE(1.0), xmin));
                ymin = max(TO_INPUT0_TYPE(0.0), min(TO_INPUT0_TYPE(1.0), ymin));
                xmax = max(TO_INPUT0_TYPE(0.0), min(TO_INPUT0_TYPE(1.0), xmax));
                ymax = max(TO_INPUT0_TYPE(0.0), min(TO_INPUT0_TYPE(1.0), ymax));
            }
            vstore4((OUTPUT_TYPE4)(xmin, ymin, xmax, ymax), 0, output + idx * OUTPUT_ROW_SIZE + 3);
            // increase starting point for the next detection in class
            buffer1[scores_size_offset + score_info.classId]++;
        }
    } else {
        const int startIdx = FUNC_CALL(get_start_idx)(buffer1, batchId, 0);
        int outputIdx = 0;
        for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++) {
            int scores_size_offset = batchId * NUM_CLASSES_ACC + idx_class;
            const int acc_num = buffer1[scores_size_offset];
            __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer0[(batchId * NUM_CLASSES + idx_class) * BUFFER_STRIDE];
            for (uint idx_score = 0; idx_score < acc_num; idx_score++) {
                SCORES_INFO score_info = scoresList[idx_score];
                const int idx = startIdx + outputIdx;
                output[idx * OUTPUT_ROW_SIZE] = TO_OUTPUT_TYPE(batchId);
                output[idx * OUTPUT_ROW_SIZE + 1] = TO_OUTPUT_TYPE((DECREASE_LABEL_ID) ? (int)score_info.classId - 1 : (int)score_info.classId);
                output[idx * OUTPUT_ROW_SIZE + 2] = TO_OUTPUT_TYPE(score_info.score);
                INPUT0_TYPE decoded_bbox[4];
                const uint loc_label = ((SHARE_LOCATION)? 0 : (int)score_info.classId);
                FUNC_CALL(get_decoded_bbox)(decoded_bbox, input_location, input_prior_box, score_info.boxId, loc_label, batchId);
                INPUT0_TYPE xmin = decoded_bbox[0];
                INPUT0_TYPE ymin = decoded_bbox[1];
                INPUT0_TYPE xmax = decoded_bbox[2];
                INPUT0_TYPE ymax = decoded_bbox[3];
                if (CLIP_AFTER_NMS) {
                    xmin = max(TO_INPUT0_TYPE(0.0), min(TO_INPUT0_TYPE(1.0), xmin));
                    ymin = max(TO_INPUT0_TYPE(0.0), min(TO_INPUT0_TYPE(1.0), ymin));
                    xmax = max(TO_INPUT0_TYPE(0.0), min(TO_INPUT0_TYPE(1.0), xmax));
                    ymax = max(TO_INPUT0_TYPE(0.0), min(TO_INPUT0_TYPE(1.0), ymax));
                }
                vstore4((OUTPUT_TYPE4)(xmin, ymin, xmax, ymax), 0, output + idx * OUTPUT_ROW_SIZE + 3);
                outputIdx++;
            }
        }
    }

    if(batchId == 0) {
        const int final_detections = FUNC_CALL(get_final_detections)(buffer1);
        unroll_for (uint i = final_detections; i < NUM_OF_IMAGES * KEEP_TOP_K; i++) {
            output[i * OUTPUT_ROW_SIZE] = (i == final_detections ? -1.0 : 0.0);
            vstore4((OUTPUT_TYPE4)(0.0, 0.0, 0.0, 0.0), 0, output + i * OUTPUT_ROW_SIZE + 1);
            vstore2((OUTPUT_TYPE2)(0.0, 0.0), 0, output + i * OUTPUT_ROW_SIZE + 5);
        }
    }
}
#endif  /* DO_STAGE_3_CAFFE */

#ifdef DO_STAGE_3_MXNET
KERNEL (detection_output_stage_final_mxnet)(__global INPUT0_TYPE* input_location,
                                            __global INPUT2_TYPE* input_prior_box,
                                            __global OUTPUT_TYPE* output,
                                            __global uchar *buffer0,
                                            __global uchar *buffer1,
                                            __global int *buffer2) {
    // Create an auxiliary stack for QuickSort
    __local int stack[QUICK_SORT_STACK_SIZE];

    for (uint idx_image = 0; idx_image < NUM_OF_IMAGES; idx_image++) {
        __global SCORES_INFO *scoresList = (__global SCORES_INFO*)&buffer0[idx_image * BUFFER_STRIDE];
        const int total_det = buffer2[idx_image * NUM_CLASSES_ACC + NUM_CLASSES];

        if (KEEP_TOP_K > -1 && total_det > KEEP_TOP_K) {
            int num_det = 0;
            for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++) {
                int scores_size_offset = idx_image * NUM_CLASSES_ACC + idx_class;
                const int acc_num = buffer2[scores_size_offset];
                __global SCORES_INFO *selectedScoresList = (__global SCORES_INFO*)&buffer1[(idx_image * NUM_CLASSES + idx_class) * BUFFER_STRIDE];

                for (uint idx_score = 0; idx_score < acc_num; idx_score++) {
                    scoresList[num_det + idx_score] = selectedScoresList[idx_score];
                }
                num_det += acc_num;
                buffer2[scores_size_offset] = 0;
            }
            FUNC_CALL(quickSortIterative)(scoresList, 0, num_det - 1, stack, true);

            for (uint idx_num_det = 0; idx_num_det < KEEP_TOP_K; idx_num_det++) {
                int scores_size_offset = idx_image * NUM_CLASSES_ACC + (int)scoresList[idx_num_det].classId;
                int acc_num = buffer2[scores_size_offset];
                __global SCORES_INFO *selectedScoresList = (__global SCORES_INFO*)&buffer1[(idx_image * NUM_CLASSES + (int)scoresList[idx_num_det].classId) * BUFFER_STRIDE];
                selectedScoresList[acc_num] = scoresList[idx_num_det];
                buffer2[scores_size_offset] = (acc_num + 1);
            }
        }
    }

    int count = 0;
    for (uint idx_image = 0; idx_image < NUM_OF_IMAGES; idx_image++) {
        for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++) {
            int scores_size_offset = idx_image * NUM_CLASSES_ACC + idx_class;
            int acc_num = buffer2[scores_size_offset];
            __global SCORES_INFO *selectedScoresList = (__global SCORES_INFO*)&buffer1[(idx_image * NUM_CLASSES + idx_class) * BUFFER_STRIDE];
            int loc_label = ((SHARE_LOCATION)? 0 : idx_class);
            for (uint idx_score = 0; idx_score < acc_num; idx_score++) {
                SCORES_INFO score_info;
                score_info = selectedScoresList[idx_score];
                output[count * OUTPUT_ROW_SIZE] = TO_OUTPUT_TYPE(idx_image);
                output[count * OUTPUT_ROW_SIZE + 1] = TO_OUTPUT_TYPE((DECREASE_LABEL_ID) ? (int)score_info.classId - 1 : (int)score_info.classId);
                output[count * OUTPUT_ROW_SIZE + 2] = TO_OUTPUT_TYPE(score_info.score);
                INPUT0_TYPE decoded_bbox[4];
                FUNC_CALL(get_decoded_bbox)(decoded_bbox, input_location, input_prior_box, score_info.boxId, loc_label, idx_image);
                INPUT0_TYPE xmin = decoded_bbox[0];
                INPUT0_TYPE ymin = decoded_bbox[1];
                INPUT0_TYPE xmax = decoded_bbox[2];
                INPUT0_TYPE ymax = decoded_bbox[3];

                if (CLIP_AFTER_NMS) {
                    xmin = max(TO_INPUT0_TYPE(0.0), min(TO_INPUT0_TYPE(1.0), xmin));
                    ymin = max(TO_INPUT0_TYPE(0.0), min(TO_INPUT0_TYPE(1.0), ymin));
                    xmax = max(TO_INPUT0_TYPE(0.0), min(TO_INPUT0_TYPE(1.0), xmax));
                    ymax = max(TO_INPUT0_TYPE(0.0), min(TO_INPUT0_TYPE(1.0), ymax));
                }
                vstore4((OUTPUT_TYPE4)(xmin, ymin, xmax, ymax), 0, output + count * OUTPUT_ROW_SIZE + 3);
                ++count;
            }
        }
    }

    if (count < NUM_OF_IMAGES * KEEP_TOP_K) {
        output[count * OUTPUT_ROW_SIZE] = -1.0;
    }
}
#endif  /* DO_STAGE_3_MXNET */
