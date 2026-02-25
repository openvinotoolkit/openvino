// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#if INPUT0_TYPE_SIZE == 2 //f16
    #define HALF_ONE 0.5h
#else
    #define HALF_ONE 0.5f
#endif

#define SORT_RESULT_CLASSID 0
#define SORT_RESULT_SCORE 1

#define SORTMODE_CLASS 0
#define SORTMODE_SCORE 1
#define SORTMODE_SCORE_THEN_INDEX 2
#define SORTMODE_SCORE_THEN_CLASS 3

#define MAX_CANDIDATES_PER_BATCH NUM_BOXES

#define GET_SELECTED_INDICES_INDEX(b, f, y, x) OUTPUT1_GET_INDEX(b, f, y, x)
#define GET_SELECTED_NUM_INDEX(b, f, y, x) OUTPUT2_GET_INDEX(b, f, y, x)

typedef struct __attribute__((__packed__)) {
    INPUT0_TYPE score;
    INPUT0_TYPE xmin;
    INPUT0_TYPE ymin;
    INPUT0_TYPE xmax;
    INPUT0_TYPE ymax;
    uint class_idx;
    uint batch_idx;
    uint index;
} FUNC(BOX_INFO);

#define BoxInfo FUNC(BOX_INFO)

#if defined(MULTICLASSNMS_STAGE_0) || defined( MULTICLASSNMS_STAGE_1)
inline void FUNC(swap_info)(__global BoxInfo* a, __global BoxInfo* b) {
    const BoxInfo temp = *a;
    *a = *b;
    *b = temp;
}

inline int FUNC(partition)(__global BoxInfo* arr, int l, int h, int sortMode) {
    const BoxInfo pivot = arr[h];

    int i = (l - 1);
    for (int j = l; j <= h - 1; j++) {
        switch(sortMode) {
            case SORTMODE_CLASS: {
                if ((arr[j].class_idx < pivot.class_idx) ||
                    (arr[j].class_idx == pivot.class_idx && arr[j].batch_idx < pivot.batch_idx) ||
                    (arr[j].class_idx == pivot.class_idx && arr[j].batch_idx == pivot.batch_idx &&
                     arr[j].score > pivot.score) ||
                    (arr[j].class_idx == pivot.class_idx && arr[j].batch_idx == pivot.batch_idx &&
                     arr[j].score == pivot.score && arr[j].index < pivot.index)) {
                    i++;
                    FUNC_CALL(swap_info)(&arr[i], &arr[j]);
                }
                break;
            }
            case SORTMODE_SCORE: {
                if ((arr[j].score > pivot.score) ||
                    (arr[j].score == pivot.score && arr[j].batch_idx < pivot.batch_idx) ||
                    (arr[j].score == pivot.score && arr[j].batch_idx == pivot.batch_idx &&
                     arr[j].class_idx < pivot.class_idx) ||
                    (arr[j].score == pivot.score && arr[j].batch_idx == pivot.batch_idx &&
                     arr[j].class_idx == pivot.class_idx && arr[j].index < pivot.index)) {
                    i++;
                    FUNC_CALL(swap_info)(&arr[i], &arr[j]);
                }
                break;
            }
            case SORTMODE_SCORE_THEN_INDEX: {
                if ((arr[j].score > pivot.score) || (arr[j].score == pivot.score && arr[j].index < pivot.index) ||
                    (arr[j].score == pivot.score && arr[j].index == pivot.index &&
                     arr[j].class_idx > pivot.class_idx) ||
                    (arr[j].score == pivot.score && arr[j].index == pivot.index &&
                     arr[j].class_idx == pivot.class_idx && arr[j].batch_idx > pivot.batch_idx)) {
                    i++;
                    FUNC_CALL(swap_info)(&arr[i], &arr[j]);
                }
                break;
            }
            case SORTMODE_SCORE_THEN_CLASS: {
                if ( (arr[j].batch_idx == pivot.batch_idx) &&
                     ((arr[j].score > pivot.score) || (arr[j].score == pivot.score && arr[j].class_idx < pivot.class_idx) ||
                     (arr[j].score == pivot.score && arr[j].class_idx == pivot.class_idx && arr[j].index < pivot.index))) {
                    i++;
                    FUNC_CALL(swap_info)(&arr[i], &arr[j]);
                }
                break;
            }
        } // switch
    }
    FUNC_CALL(swap_info)(&arr[i + 1], &arr[h]);
    return (i + 1);
}

inline void FUNC(bubbleSortIterative)(__global BoxInfo* arr, int l, int h, int sortMode) {
    for (int i = 0; i < h - l; i++) {
        bool swapped = false;
        for (int j = l; j < h - i; j++) {
            switch(sortMode) {
                case SORTMODE_CLASS: {
                    if ((arr[j].class_idx < arr[j + 1].class_idx) ||
                        (arr[j].class_idx == arr[j + 1].class_idx && arr[j].batch_idx < arr[j + 1].batch_idx) ||
                        (arr[j].class_idx == arr[j + 1].class_idx && arr[j].batch_idx == arr[j + 1].batch_idx &&
                         arr[j].score > arr[j + 1].score) ||
                        (arr[j].class_idx == arr[j + 1].class_idx && arr[j].batch_idx == arr[j + 1].batch_idx &&
                         arr[j].score == arr[j + 1].score && arr[j].index < arr[j + 1].index)) {
                        FUNC_CALL(swap_info)(&arr[j], &arr[j + 1]);
                        swapped = true;
                    }
                    break;
                }
                case SORTMODE_SCORE: {
                    if ((arr[j].score > arr[j + 1].score) ||
                        (arr[j].score == arr[j + 1].score && arr[j].batch_idx < arr[j + 1].batch_idx) ||
                        (arr[j].score == arr[j + 1].score && arr[j].batch_idx == arr[j + 1].batch_idx &&
                         arr[j].class_idx < arr[j + 1].class_idx) ||
                        (arr[j].score == arr[j + 1].score && arr[j].batch_idx == arr[j + 1].batch_idx &&
                         arr[j].class_idx == arr[j + 1].class_idx && arr[j].index < arr[j + 1].index)) {
                        FUNC_CALL(swap_info)(&arr[j], &arr[j + 1]);
                        swapped = true;
                    }
                    break;
                }
                case SORTMODE_SCORE_THEN_INDEX: {
                    if ((arr[j].score > arr[j + 1].score) ||
                        (arr[j].score == arr[j + 1].score && arr[j].index < arr[j + 1].index) ||
                        (arr[j].score == arr[j + 1].score && arr[j].index == arr[j + 1].index &&
                         arr[j].class_idx < arr[j + 1].class_idx) ||
                        (arr[j].score == arr[j + 1].score && arr[j].index == arr[j + 1].index &&
                         arr[j].class_idx == arr[j + 1].class_idx && arr[j].batch_idx < arr[j + 1].batch_idx)) {
                        FUNC_CALL(swap_info)(&arr[j], &arr[j + 1]);
                        swapped = true;
                    }
                    break;
                }
                case SORTMODE_SCORE_THEN_CLASS: {
                    if ( (arr[j].batch_idx == arr[j + 1].batch_idx) &&
                         ((arr[j].score > arr[j + 1].score) || (arr[j].score == arr[j + 1].score && arr[j].class_idx < arr[j + 1].class_idx) ||
                         (arr[j].score == arr[j + 1].score && arr[j].class_idx == arr[j + 1].class_idx && arr[j].index < arr[j + 1].index))) {
                        FUNC_CALL(swap_info)(&arr[j], &arr[j + 1]);
                        swapped = true;
                    }
                    break;
                }
            } // switch
        }

        if (!swapped)
            break;
    }
}

inline void FUNC(quickSortIterative)(__global BoxInfo* arr, int l, int h, int sortMode) {
    if (l == h || l < 0 || h <= 0) {
        return;
    }
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
        int p = FUNC_CALL(partition)(arr, l, h, sortMode);

        // If there are elements on left side of pivot,
        // then push left side to stack
        if (p - 1 > l) {
            if (top >= (kStackSize - 1)) {
                FUNC_CALL(bubbleSortIterative)(arr, l, p - 1, sortMode);
            } else {
                stack[++top] = l;
                stack[++top] = p - 1;
            }
        }

        // If there are elements on right side of pivot,
        // then push right side to stack
        if (p + 1 < h) {
            if (top >= (kStackSize - 1)) {
                FUNC_CALL(bubbleSortIterative)(arr, p + 1, h, sortMode);
            } else {
                stack[++top] = p + 1;
                stack[++top] = h;
            }
        }
    }
}
#endif

#ifdef MULTICLASSNMS_STAGE_0
inline INPUT0_TYPE FUNC(intersectionOverUnion)(const __global BoxInfo* i, const __global BoxInfo* j) {
    const INPUT0_TYPE norm = !NORMALIZED;

    INPUT0_TYPE areaI = (i->ymax - i->ymin + norm) * (i->xmax - i->xmin + norm);
    INPUT0_TYPE areaJ = (j->ymax - j->ymin + norm) * (j->xmax - j->xmin + norm);

    if (areaI <= INPUT0_VAL_ZERO || areaJ <= INPUT0_VAL_ZERO) {
        return INPUT0_VAL_ZERO;
    }

    const float intersection_ymin = max(i->ymin, j->ymin);
    const float intersection_xmin = max(i->xmin, j->xmin);
    const float intersection_ymax = min(i->ymax, j->ymax);
    const float intersection_xmax = min(i->xmax, j->xmax);

    const float intersection_area = max(intersection_ymax - intersection_ymin + norm, 0.0f) *
                              max(intersection_xmax - intersection_xmin + norm, 0.0f);

    return intersection_area / (areaI + areaJ - intersection_area);
}

inline uint FUNC(nms)(const __global INPUT0_TYPE* boxes,
                                     const __global INPUT0_TYPE* scores,
                                     const uint batch_idx,
                                     const uint class_idx,
                                     const uint num_boxes,
                                     const uint num_previous_boxes,
                                     __global BoxInfo* box_info) {
    uint candidates_num = 0;

    for (uint box_idx = 0; box_idx < num_boxes; ++box_idx) {
        const uint box_number = num_previous_boxes + box_idx;

        #ifdef HAS_ROISNUM
            const uint score_idx = INPUT1_GET_INDEX(class_idx, box_number, 0, 0);
       #else
            const uint score_idx = INPUT1_GET_INDEX(batch_idx, class_idx, box_idx, 0);
        #endif
        INPUT0_TYPE score = scores[score_idx];

        if (score < SCORE_THRESHOLD) {
            continue;
        }

        __global BoxInfo* candidate_box = box_info + candidates_num;
        candidate_box->class_idx = class_idx;
        candidate_box->batch_idx = batch_idx;
        candidate_box->index = box_idx;
        candidate_box->score = score;

        #ifdef HAS_ROISNUM
            const uint first_dim = class_idx;
        #else
            const uint first_dim = batch_idx;
        #endif

        candidate_box->xmin = boxes[INPUT0_GET_INDEX(first_dim, box_number, 0, 0)];
        candidate_box->ymin = boxes[INPUT0_GET_INDEX(first_dim, box_number, 1, 0)];
        candidate_box->xmax = boxes[INPUT0_GET_INDEX(first_dim, box_number, 2, 0)];
        candidate_box->ymax = boxes[INPUT0_GET_INDEX(first_dim, box_number, 3, 0)];

        ++candidates_num;
    }

    if (candidates_num == 0) {  // early drop
        return candidates_num;
    }

    // sort by score in current class - must be higher score/lower index first (see std::greater<BoxInfo> in ref impl)
    FUNC_CALL(quickSortIterative)(box_info, 0, candidates_num - 1, SORTMODE_SCORE_THEN_INDEX);

    // threshold nms_top_k for each class
    if (NMS_TOP_K > -1 && NMS_TOP_K < candidates_num) {
        candidates_num = NMS_TOP_K;
    }

    if (candidates_num <= 0) {  // early drop
        return candidates_num;  // empty
    }

    INPUT0_TYPE adaptive_threshold = IOU_THRESHOLD;
    uint selected_size = 0;
    for (size_t i = 0; i < candidates_num; ++i) {
        __global BoxInfo* next_candidate = box_info + i;

        bool should_hard_suppress = false;

        if (NMS_ETA < 1 && adaptive_threshold > HALF_ONE) {
            adaptive_threshold *= NMS_ETA;
        }

        for (uint j = 0; j < selected_size; ++j) {
            __global BoxInfo* selected = box_info + j;
            const float iou = FUNC_CALL(intersectionOverUnion)(box_info + i, box_info + j);

            if (iou >= adaptive_threshold) {
                should_hard_suppress = true;
            }
        }
        if (!should_hard_suppress) {
            box_info[selected_size] = box_info[i];
            ++selected_size;
        }
    }

    return selected_size;
}

inline uint FUNC(multiclass_nms)(const __global INPUT0_TYPE* boxes,
                                 const __global INPUT0_TYPE* scores,
                                 const uint num_boxes,
                                 const uint num_previous_boxes,
                                 const uint batch_idx,
                                 __global BoxInfo* box_info) {
    uint detection_count = 0;

    for (uint class_idx = 0; class_idx < NUM_CLASSES; ++class_idx) {
        if (class_idx == BACKGROUND_CLASS) {
            continue;
        }

        const uint detected = FUNC_CALL(nms)(boxes, scores, batch_idx, class_idx, num_boxes, num_previous_boxes, box_info + detection_count);
        detection_count += detected;
    }

    FUNC_CALL(quickSortIterative)(box_info, 0, detection_count - 1, SORTMODE_SCORE_THEN_CLASS);

    if (KEEP_TOP_K > -1 && KEEP_TOP_K < detection_count) {
        detection_count = KEEP_TOP_K;
    }


#if !(SORT_RESULT_ACROSS_BATCH) && (SORT_RESULT_TYPE == SORT_RESULT_CLASSID)
    FUNC_CALL(quickSortIterative)(box_info, 0, detection_count - 1, SORTMODE_CLASS);
#endif

    return detection_count;
}

KERNEL(multiclass_nms_ref_stage_0)(
    const __global INPUT0_TYPE* boxes,
    const __global INPUT1_TYPE* scores,
#ifdef HAS_ROISNUM
    const __global INPUT2_TYPE* roisnum,
#endif
    __global BoxInfo* box_info,
    __global OUTPUT_TYPE* selected_outputs,
    __global OUTPUT1_TYPE* selected_indices,
    __global OUTPUT2_TYPE* selected_num) {

    const uint batch_idx = get_global_id(0);
    const uint box_info_offset = batch_idx * MAX_CANDIDATES_PER_BATCH;

    uint num_previous_boxes = 0;
    #ifdef HAS_ROISNUM
        const uint num_boxes = roisnum[INPUT2_GET_INDEX(batch_idx, 0, 0, 0)];
        if (num_boxes <= 0) {
            selected_num[GET_SELECTED_NUM_INDEX(batch_idx, 0, 0, 0)] = 0;
            return;
        }

        if (batch_idx > 0) {
            for (uint i = 0; i < batch_idx; ++i) {
                num_previous_boxes += roisnum[INPUT2_GET_INDEX(i, 0, 0, 0)];
            }
        }
    #else
        const uint num_boxes = NUM_BOXES;
    #endif

    const uint nselected = FUNC_CALL(multiclass_nms)(boxes, scores, num_boxes, num_previous_boxes, batch_idx, box_info + box_info_offset);

    selected_num[GET_SELECTED_NUM_INDEX(batch_idx, 0, 0, 0)] = nselected;
}

#endif //MULTICLASSNMS_STAGE_0

#ifdef MULTICLASSNMS_STAGE_1
KERNEL(multiclass_nms_ref_stage_1)(
    const __global INPUT0_TYPE* boxes,
    const __global INPUT1_TYPE* scores,
#ifdef HAS_ROISNUM
    const __global INPUT2_TYPE* roisnum,
#endif
    __global BoxInfo* box_info,
    __global OUTPUT_TYPE* selected_outputs,
    __global OUTPUT1_TYPE* selected_indices,
    __global OUTPUT2_TYPE* selected_num) {

    // pack box_infos
    uint dst_offset = selected_num[GET_SELECTED_NUM_INDEX(0, 0, 0, 0)];
    for(uint batch_idx = 0; batch_idx < NUM_BATCHES - 1; ++batch_idx) {
        const uint boxes_to_copy = selected_num[GET_SELECTED_NUM_INDEX(batch_idx + 1, 0, 0, 0)];
        const uint src_offset = (batch_idx + 1) * MAX_CANDIDATES_PER_BATCH;

        for(uint i = 0; i < boxes_to_copy; ++i) {
            box_info[dst_offset + i] = box_info[src_offset + i];
        }

        dst_offset += boxes_to_copy;
    }

#if SORT_RESULT_ACROSS_BATCH
    #if SORT_RESULT_TYPE == SORT_RESULT_SCORE
        FUNC_CALL(quickSortIterative)(box_info, 0, dst_offset - 1, SORTMODE_SCORE);
    #elif SORT_RESULT_TYPE == SORT_RESULT_CLASSID
        FUNC_CALL(quickSortIterative)(box_info, 0, dst_offset - 1, SORTMODE_CLASS);
    #endif
#endif
}
#endif //MULTICLASSNMS_STAGE_1

#ifdef MULTICLASSNMS_STAGE_2
KERNEL(multiclass_nms_ref_stage_2)(
    const __global INPUT0_TYPE* boxes,
    const __global INPUT1_TYPE* scores,
#ifdef HAS_ROISNUM
    const __global INPUT2_TYPE* roisnum,
#endif
    __global BoxInfo* box_info,
    __global OUTPUT_TYPE* selected_outputs,
    __global OUTPUT1_TYPE* selected_indices,
    __global OUTPUT2_TYPE* selected_num) {

    // fill outputs
    const uint batch_idx = get_global_id(0);

    uint box_info_offset = 0;
    for (uint i = 0; i < batch_idx; ++i) {
        box_info_offset += selected_num[GET_SELECTED_NUM_INDEX(i, 0, 0, 0)];
    }

    const uint nselected = selected_num[GET_SELECTED_NUM_INDEX(batch_idx, 0, 0, 0)];

    uint idx;
    for (idx = 0; idx < nselected; ++idx) {
        const __global BoxInfo* info = box_info + box_info_offset + idx;

        const uint box_number = batch_idx * MAX_OUTPUT_BOXES_PER_BATCH + idx;
        selected_outputs[OUTPUT_GET_INDEX(box_number, 0, 0, 0)] = (OUTPUT_TYPE)info->class_idx;
        selected_outputs[OUTPUT_GET_INDEX(box_number, 1, 0, 0)] = info->score;
        selected_outputs[OUTPUT_GET_INDEX(box_number, 2, 0, 0)] = info->xmin;
        selected_outputs[OUTPUT_GET_INDEX(box_number, 3, 0, 0)] = info->ymin;
        selected_outputs[OUTPUT_GET_INDEX(box_number, 4, 0, 0)] = info->xmax;
        selected_outputs[OUTPUT_GET_INDEX(box_number, 5, 0, 0)] = info->ymax;

        #ifdef HAS_ROISNUM
            const uint num_boxes = roisnum[INPUT2_GET_INDEX(batch_idx, 0, 0, 0)];
            uint offset = 0;
            for (uint i = 0; i < info->batch_idx; ++i) {
                offset += roisnum[INPUT2_GET_INDEX(i, 0, 0, 0)];
            }

            selected_indices[GET_SELECTED_INDICES_INDEX(box_number, 0, 0, 0)] = (offset + info->index) * NUM_CLASSES + info->class_idx;
        #else
            selected_indices[GET_SELECTED_INDICES_INDEX(box_number, 0, 0, 0)] = info->batch_idx * NUM_BOXES + info->index;
        #endif
    }

    // tail
    for (; idx < MAX_OUTPUT_BOXES_PER_BATCH; ++idx) {
        const uint box_number = batch_idx * MAX_OUTPUT_BOXES_PER_BATCH + idx;

        selected_outputs[OUTPUT_GET_INDEX(box_number, 0, 0, 0)] = -1;
        selected_outputs[OUTPUT_GET_INDEX(box_number, 1, 0, 0)] = -1;
        selected_outputs[OUTPUT_GET_INDEX(box_number, 2, 0, 0)] = -1;
        selected_outputs[OUTPUT_GET_INDEX(box_number, 3, 0, 0)] = -1;
        selected_outputs[OUTPUT_GET_INDEX(box_number, 4, 0, 0)] = -1;
        selected_outputs[OUTPUT_GET_INDEX(box_number, 5, 0, 0)] = -1;

        selected_indices[GET_SELECTED_INDICES_INDEX(box_number, 0, 0, 0)] = -1;
    }
}
#endif //MULTICLASSNMS_STAGE_2

#undef MAX_CANDIDATES_PER_BATCH
#undef SORT_RESULT_SCORE
#undef SORT_RESULT_CLASSID
#undef SORTMODE_SCORE_THEN_CLASS
#undef SORTMODE_SCORE_THEN_INDEX
#undef SORTMODE_SCORE
#undef SORTMODE_CLASS
#undef HALF_ONE
