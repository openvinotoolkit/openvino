// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

#define NUM_BATCHES INPUT0_BATCH_NUM
#define NUM_BOXES   INPUT0_FEATURE_NUM
#define NUM_CLASSES INPUT1_FEATURE_NUM

typedef struct {
    int batch_idx;
    int class_idx;
    int box_idx;
    INPUT1_TYPE score;
} FUNC(BoxInfo);

#define BOX_INFO FUNC(BoxInfo)

inline INPUT1_TYPE FUNC(decay_gaussian)(INPUT1_TYPE iou, INPUT1_TYPE max_iou) {
    return exp((max_iou * max_iou - iou * iou) * GAUSSIAN_SIGMA);
}

inline INPUT1_TYPE FUNC(decay_linear)(INPUT1_TYPE iou, INPUT1_TYPE max_iou) {
    return (INPUT1_VAL_ONE - iou) / (INPUT1_VAL_ONE - max_iou + TINY);
}

inline void FUNC(swap)(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

inline void FUNC(sortIterative)(const __global INPUT1_TYPE* scores,
                                      const int batchId,
                                      const int classId,
                                      int* indices,
                                      const int size) {
    for (int i = 1; i <= size; i++) {
        bool swapped = false;
        for (int j = 0; j < size - i; j++) {
            const INPUT1_TYPE score_curr = scores[INPUT1_GET_INDEX(batchId, classId, 0, indices[j])];
            const INPUT1_TYPE score_next = scores[INPUT1_GET_INDEX(batchId, classId, 0, indices[j + 1])];
            if (score_curr < score_next) {
                FUNC_CALL(swap)(&indices[j], &indices[j + 1]);
                swapped = true;
            }
        }

        if (!swapped)
            break;
    }
}

inline void FUNC(swap_boxes)(__global BOX_INFO* a, __global BOX_INFO* b) {
    BOX_INFO temp = *a;
    *a = *b;
    *b = temp;
}

inline void FUNC(sortIterativeBoxes)(__global BOX_INFO* boxes, int l, int h) {
    for (int i = 1; i < h - l; i++) {
        bool swapped = false;
        for (int j = l; j < h - i; j++) {
            if ((boxes[j].score < boxes[j + 1].score) ||
                (boxes[j].score == boxes[j + 1].score && boxes[j].class_idx > boxes[j + 1].class_idx) ||
                (boxes[j].score == boxes[j + 1].score && boxes[j].class_idx == boxes[j + 1].class_idx &&
                 boxes[j].box_idx > boxes[j + 1].box_idx)) {
                FUNC_CALL(swap_boxes)(&boxes[j], &boxes[j + 1]);
                swapped = true;
            }
        }

        if (!swapped)
            break;
    }
}

inline void FUNC(sortIterativeBoxesAcrossBatches)(__global BOX_INFO* boxes) {
    const int size = NUM_BATCHES * NUM_CLASSES * MAX_BOXES_PER_CLASS;
    for (int i = 1; i < size; i++) {
        bool swapped = false;
        for (int j = 0; j < size - i; j++) {
            __global BOX_INFO* l = boxes + j;
            __global BOX_INFO* r = boxes + j + 1;
// sort by score
#if SORT_TYPE == 1
            if ((l->score < r->score) || (l->score == r->score && l->batch_idx > r->batch_idx) ||
                (l->score == r->score && l->batch_idx == r->batch_idx && l->class_idx > r->class_idx) ||
                (l->score == r->score && l->batch_idx == r->batch_idx && l->class_idx == r->class_idx &&
                 l->box_idx > r->box_idx)) {
                FUNC_CALL(swap_boxes)(l, r);
                swapped = true;
            }
// sort by class id
#elif SORT_TYPE == 0
            if (r->score != INPUT1_VAL_ZERO &&
                ((l->score == INPUT1_VAL_ZERO) ||  // case with empty buffer
                 (l->class_idx > r->class_idx) || (l->class_idx == r->class_idx && l->batch_idx > r->batch_idx) ||
                 (l->class_idx == r->class_idx && l->batch_idx == r->batch_idx && l->score < r->score) ||
                 (l->class_idx == r->class_idx && l->batch_idx == r->batch_idx && l->score == r->score &&
                  l->box_idx > r->box_idx))) {
                FUNC_CALL(swap_boxes)(l, r);
                swapped = true;
            }
#endif
        }

        if (!swapped)
            break;
    }
}

inline COORD_TYPE_4 FUNC(getBoxCoords)(const __global INPUT0_TYPE* boxes, const short batch, const ushort box_idx) {
    COORD_TYPE_4 coords = (COORD_TYPE_4)(boxes[INPUT0_GET_INDEX(batch, box_idx, 0, 0)],
                                         boxes[INPUT0_GET_INDEX(batch, box_idx, 0, 1)],
                                         boxes[INPUT0_GET_INDEX(batch, box_idx, 0, 2)],
                                         boxes[INPUT0_GET_INDEX(batch, box_idx, 0, 3)]);

    // uncomment when flipped coordinates will be fixed in reference impl
    /*
    const INPUT0_TYPE x1 = min(coords[0], coords[2]);
    const INPUT0_TYPE x2 = max(coords[0], coords[2]);
    const INPUT0_TYPE y1 = min(coords[1], coords[3]);
    const INPUT0_TYPE y2 = max(coords[1], coords[3]);
    coords[0] = x1;
    coords[1] = y1;
    coords[2] = x2;
    coords[3] = y2;
    */

    return coords;
}

inline INPUT0_TYPE FUNC(area)(const INPUT0_TYPE w, const INPUT0_TYPE h) {
    return (w + NORM) * (h + NORM);
}

inline INPUT0_TYPE FUNC(areaBox)(const COORD_TYPE_4 box) {
    if (box[2] < box[0] || box[3] < box[1])
        return INPUT0_VAL_ZERO;
    return FUNC_CALL(area)(box[3] - box[1], box[2] - box[0]);
}

inline INPUT0_TYPE FUNC(intersectionOverUnion)(const COORD_TYPE_4 box1, const COORD_TYPE_4 box2) {
    if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] || box2[3] < box1[1])
        return INPUT0_VAL_ZERO;

    const INPUT0_TYPE area = FUNC_CALL(areaBox)(box1);
    const INPUT0_TYPE areaBox = FUNC_CALL(areaBox)(box2);

    const INPUT0_TYPE intersection_xmin = max(box1[0], box2[0]);
    const INPUT0_TYPE intersection_ymin = max(box1[1], box2[1]);
    const INPUT0_TYPE intersection_xmax = min(box1[2], box2[2]);
    const INPUT0_TYPE intersection_ymax = min(box1[3], box2[3]);

    const INPUT0_TYPE intersection_area =
        FUNC_CALL(area)(intersection_xmax - intersection_xmin, intersection_ymax - intersection_ymin);
    const INPUT0_TYPE union_area = area + areaBox - intersection_area;

    return intersection_area / union_area;
}

#ifdef MATRIX_NMS_STAGE_0
KERNEL(matrix_nms_ref_stage_0)
(const __global INPUT0_TYPE* input_boxes,
 const __global INPUT1_TYPE* input_scores,
 __global uchar* buffer0,
 __global int* selected_boxes_num) {
    const int batchId = get_global_id(0);
    const int classId = get_global_id(1);

    if (classId == BACKGROUND_CLASS)
        return;

    int sorted_score_indices[NUM_BOXES];

    for (int i = 0; i < NUM_BOXES; ++i)
        sorted_score_indices[i] = i;

    int valid_boxes_num = 0;
    for (int i = 0; i < NUM_BOXES; i++) {
        if (input_scores[INPUT1_GET_INDEX(batchId, classId, 0, i)] > SCORE_THRESHOLD)
            ++valid_boxes_num;
    }

    // TODO: consider faster sorting algorithm
    FUNC_CALL(sortIterative)(input_scores, batchId, classId, sorted_score_indices, NUM_BOXES);

    valid_boxes_num = min(valid_boxes_num, MAX_BOXES_PER_CLASS);

    const int matrix_size = MAX_BOXES_PER_CLASS < 3 ? 1 : (MAX_BOXES_PER_CLASS * (MAX_BOXES_PER_CLASS - 1)) >> 1;
    INPUT1_TYPE iou_matrix[matrix_size];
    INPUT1_TYPE iou_max[MAX_BOXES_PER_CLASS];

    iou_max[0] = INPUT1_VAL_ZERO;
    for (int i = 1; i < valid_boxes_num; ++i) {
        INPUT1_TYPE max_iou = INPUT1_VAL_ZERO;
        const COORD_TYPE_4 box_i = FUNC_CALL(getBoxCoords)(input_boxes, batchId, sorted_score_indices[i]);
        for (int j = 0; j < i; ++j) {
            const COORD_TYPE_4 box_j = FUNC_CALL(getBoxCoords)(input_boxes, batchId, sorted_score_indices[j]);
            const INPUT1_TYPE iou = FUNC_CALL(intersectionOverUnion)(box_i, box_j);

            max_iou = max(iou, max_iou);
            iou_matrix[i * (i - 1) / 2 + j] = iou;
        }
        iou_max[i] = max_iou;
    }

    const INPUT1_TYPE first_score = input_scores[INPUT1_GET_INDEX(batchId, classId, 0, sorted_score_indices[0])];

    __global BOX_INFO* box_info = (__global BOX_INFO*)buffer0;
    box_info = &box_info[batchId * NUM_CLASSES * MAX_BOXES_PER_CLASS + classId * MAX_BOXES_PER_CLASS];

    int box_info_counter = 0;
    if (first_score > POST_THRESHOLD && valid_boxes_num > 0) {
        box_info[box_info_counter].class_idx = classId;
        box_info[box_info_counter].score = first_score;
        box_info[box_info_counter].box_idx = sorted_score_indices[0];
        box_info[box_info_counter].batch_idx = batchId;
        ++box_info_counter;
    }

    for (int i = 1; i < valid_boxes_num; ++i) {
        INPUT1_TYPE min_decay = INPUT1_VAL_ONE;
        for (int j = 0; j < i; ++j) {
            INPUT1_TYPE iou = iou_matrix[i * (i - 1) / 2 + j];
            INPUT1_TYPE decay =
                DECAY_FUNC == 0 ? FUNC_CALL(decay_gaussian)(iou, iou_max[j]) : FUNC_CALL(decay_linear)(iou, iou_max[j]);
            min_decay = min(min_decay, decay);
        }

        INPUT1_TYPE ds = min_decay * input_scores[INPUT1_GET_INDEX(batchId, classId, 0, sorted_score_indices[i])];

        if (ds <= POST_THRESHOLD)
            continue;

        box_info[box_info_counter].batch_idx = batchId;
        box_info[box_info_counter].class_idx = classId;
        box_info[box_info_counter].box_idx = sorted_score_indices[i];
        box_info[box_info_counter].score = ds;
        ++box_info_counter;
    }

    selected_boxes_num[batchId * NUM_CLASSES + classId] = box_info_counter;
}
#endif /* MATRIX_NMS_STAGE_0 */

#ifdef MATRIX_NMS_STAGE_1
KERNEL(matrix_nms_ref_stage_1)
(__global OUTPUT2_TYPE* valid_outputs, __global uchar* buffer0, __global int* selected_boxes_num) {
    const int batchId = get_global_id(0);

    __global BOX_INFO* box_info = (__global BOX_INFO*)buffer0;

    const int first_idx = batchId * NUM_CLASSES * MAX_BOXES_PER_CLASS;
    const int last_idx = first_idx + NUM_CLASSES * MAX_BOXES_PER_CLASS;

    // TODO: consider faster sorting algorithm
    FUNC_CALL(sortIterativeBoxes)(box_info, first_idx, last_idx);

    for (int i = 0; i < NUM_CLASSES; ++i) {
        if (i == BACKGROUND_CLASS)
            continue;

        valid_outputs[OUTPUT2_GET_INDEX(batchId, 0, 0, 0)] += selected_boxes_num[batchId * NUM_CLASSES + i];
    }
}
#endif /* MATRIX_NMS_STAGE_1 */

#ifdef MATRIX_NMS_STAGE_2
KERNEL(matrix_nms_ref_stage_2)
(const __global INPUT0_TYPE* input_boxes,
 __global OUTPUT_TYPE* output,
 __global OUTPUT1_TYPE* selected_indices,
 __global OUTPUT2_TYPE* valid_outputs,
 __global uchar* buffer0) {
    __global BOX_INFO* box_info = (__global BOX_INFO*)buffer0;

    // TODO: consider faster sorting algorithm
    // and index sorting instead of data sorting
#if SORT_RESULT_ACROSS_BATCH == 1 && SORT_TYPE != 2
    FUNC_CALL(sortIterativeBoxesAcrossBatches)(box_info);
#endif

    int output_idx = 0;
    int box_info_idx = 0;
    for (int i = 0; i < NUM_BATCHES; ++i) {
        if (KEEP_TOP_K != -1 && KEEP_TOP_K < valid_outputs[OUTPUT2_GET_INDEX(i, 0, 0, 0)])
            valid_outputs[OUTPUT2_GET_INDEX(i, 0, 0, 0)] = KEEP_TOP_K;

#if SORT_RESULT_ACROSS_BATCH == 0
        box_info_idx = i * NUM_CLASSES * MAX_BOXES_PER_CLASS;
#endif

        unroll_for(int j = 0; j < valid_outputs[OUTPUT2_GET_INDEX(i, 0, 0, 0)]; ++j) {
            output[OUTPUT_GET_INDEX(output_idx, 0, 0, 0)] = box_info[box_info_idx].class_idx;
            output[OUTPUT_GET_INDEX(output_idx, 1, 0, 0)] = box_info[box_info_idx].score;
            output[OUTPUT_GET_INDEX(output_idx, 2, 0, 0)] =
                input_boxes[INPUT0_GET_INDEX(box_info[box_info_idx].batch_idx, box_info[box_info_idx].box_idx, 0, 0)];
            output[OUTPUT_GET_INDEX(output_idx, 3, 0, 0)] =
                input_boxes[INPUT0_GET_INDEX(box_info[box_info_idx].batch_idx, box_info[box_info_idx].box_idx, 0, 1)];
            output[OUTPUT_GET_INDEX(output_idx, 4, 0, 0)] =
                input_boxes[INPUT0_GET_INDEX(box_info[box_info_idx].batch_idx, box_info[box_info_idx].box_idx, 0, 2)];
            output[OUTPUT_GET_INDEX(output_idx, 5, 0, 0)] =
                input_boxes[INPUT0_GET_INDEX(box_info[box_info_idx].batch_idx, box_info[box_info_idx].box_idx, 0, 3)];

            selected_indices[OUTPUT1_GET_INDEX(output_idx, 0, 0, 0)] =
                box_info[box_info_idx].batch_idx * NUM_BOXES + box_info[box_info_idx].box_idx;

            ++output_idx;
            ++box_info_idx;
        }

        // Paddings
        while (output_idx < (i + 1) * MAX_BOXES_PER_BATCH) {
            unroll_for(int j = 0; j < 6; ++j) {
                output[OUTPUT_GET_INDEX(output_idx, j, 0, 0)] = -OUTPUT_VAL_ONE;
            }
            selected_indices[OUTPUT1_GET_INDEX(output_idx, 0, 0, 0)] = -OUTPUT1_VAL_ONE;
            ++output_idx;
        }
    }
}
#endif /* MATRIX_NMS_STAGE_2 */

#undef NUM_BATCHES
#undef NUM_BOXES
#undef NUM_CLASSES
#undef BOX_INFO
