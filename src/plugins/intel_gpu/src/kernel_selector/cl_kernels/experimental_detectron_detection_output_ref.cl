// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#define INPUT_TYPE  INPUT0_TYPE
#define INPUT_TYPE2 MAKE_VECTOR_TYPE(INPUT0_TYPE, 2)
#define INPUT_TYPE4 MAKE_VECTOR_TYPE(INPUT0_TYPE, 4)

#if INPUT0_TYPE_SIZE == 2  // f16
#    define HALF_ONE (INPUT_TYPE2)(0.5h)
#    define ZERO     0.0h
#    define ONE      1.0h
#else
#    define HALF_ONE (INPUT_TYPE2)(0.5f)
#    define ZERO     0.0f
#    define ONE      1.0f
#endif

#define ZERO2 (INPUT_TYPE2)(ZERO)
#define ZERO4 (INPUT_TYPE4)(ZERO)

#define COORDINATE_OFFSET (INPUT_TYPE2)(ONE)

#define DELTA_WEIGHTS (INPUT_TYPE4)(DELTA_WEIGHT_X, DELTA_WEIGHT_Y, DELTA_WEIGHT_LOG_W, DELTA_WEIGHT_LOG_H)

#define MAX_DELTA_LOG_SIZE (INPUT_TYPE2)(TO_INPUT0_TYPE(MAX_DELTA_LOG_WH))

typedef struct __attribute__((packed)) {
    INPUT_TYPE score __attribute__((aligned(4)));
    uint class_idx;
    uint box_idx;
} FUNC(SCI);

#define ScoreClassIndex FUNC(SCI)

inline void FUNC(swap_info)(__global ScoreClassIndex* a, __global ScoreClassIndex* b) {
    const ScoreClassIndex temp = *a;
    *a = *b;
    *b = temp;
}

inline int FUNC(partition)(__global ScoreClassIndex* arr, int l, int h) {
    const INPUT_TYPE pivot_score = arr[h].score;
    const size_t pivot_box_idx = arr[h].box_idx;
    int i = (l - 1);
    for (int j = l; j <= h - 1; j++) {
        if ((arr[j].score > pivot_score) || (arr[j].score == pivot_score && arr[j].box_idx < pivot_box_idx)) {
            i++;
            FUNC_CALL(swap_info)(&arr[i], &arr[j]);
        }
    }
    FUNC_CALL(swap_info)(&arr[i + 1], &arr[h]);
    return (i + 1);
}

inline void FUNC(bubbleSortIterative)(__global ScoreClassIndex* arr, int l, int h) {
    for (int i = 0; i < h - l; i++) {
        bool swapped = false;
        for (int j = l; j < h - i; j++) {
            if ((arr[j].score > arr[j + 1].score) ||
                (arr[j].score == arr[j + 1].score && arr[j].box_idx < arr[j + 1].box_idx)) {
                FUNC_CALL(swap_info)(&arr[j], &arr[j + 1]);
                swapped = true;
            }
        }

        if (!swapped)
            break;
    }
}

inline void FUNC(quickSortIterative)(__global ScoreClassIndex* arr, int l, int h) {
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
        int p = FUNC_CALL(partition)(arr, l, h);

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

// FIXME: rename stages accordingly
#ifdef EDDO_STAGE_0_REFINE_BOXES

// 0. Refine boxes
KERNEL(eddo_ref_stage_0)
(const __global INPUT_TYPE* boxes,
 const __global INPUT_TYPE* deltas,
 const __global INPUT_TYPE* scores,
 const __global INPUT_TYPE* im_info,
 __global INPUT_TYPE* refined_boxes,
 __global INPUT_TYPE* refined_box_areas,
 __global INPUT_TYPE* refined_scores) {
    const size_t roi_count = get_global_size(0);
    size_t roi_idx = get_global_id(0);

#ifdef CLASS_AGNOSTIC_BOX_REGRESSION
    size_t class_idx = get_global_id(1) + 1;
#else
    size_t class_idx = get_global_id(1);
#endif

#ifdef USE_BLOCKED_FORMAT
    INPUT_TYPE4 box;
    box[0] = boxes[INPUT0_GET_INDEX(roi_idx, 0, 0, 0)];
    box[1] = boxes[INPUT0_GET_INDEX(roi_idx, 1, 0, 0)];
    box[2] = boxes[INPUT0_GET_INDEX(roi_idx, 2, 0, 0)];
    box[3] = boxes[INPUT0_GET_INDEX(roi_idx, 3, 0, 0)];
#else
    INPUT_TYPE4 box = vload4(roi_idx, boxes);
#endif

    if (any(islessequal(box.hi - box.lo, ZERO2))) {
        const int refined_offset = roi_count * class_idx + roi_idx;
        refined_scores[refined_offset] = ZERO;
    } else {
        const int offset = NUM_CLASSES * roi_idx + class_idx;

        // width & height of box
        INPUT_TYPE2 box_size = (box.hi - box.lo + COORDINATE_OFFSET);

        // center location of box
        const INPUT_TYPE2 center = box.lo + HALF_ONE * box_size;

#ifdef USE_BLOCKED_FORMAT
        INPUT_TYPE4 delta;
        delta[0] = deltas[INPUT1_GET_INDEX(roi_idx, class_idx * 4, 0, 0)];
        delta[1] = deltas[INPUT1_GET_INDEX(roi_idx, class_idx * 4 + 1, 0, 0)];
        delta[2] = deltas[INPUT1_GET_INDEX(roi_idx, class_idx * 4 + 2, 0, 0)];
        delta[3] = deltas[INPUT1_GET_INDEX(roi_idx, class_idx * 4 + 3, 0, 0)];
        delta  = delta / DELTA_WEIGHTS;
#else
        const INPUT_TYPE4 delta = vload4(offset, deltas) / DELTA_WEIGHTS;
#endif

        // new center location according to deltas (dx, dy)
        const INPUT_TYPE2 new_center = delta.lo * box_size + center;
        // new width & height according to deltas d(log w), d(log h)
        const INPUT_TYPE2 new_size = exp(min(delta.hi, MAX_DELTA_LOG_SIZE)) * box_size;

        // update upper-left corner and lower-right corners respectively
        INPUT_TYPE4 new_box =
            (INPUT_TYPE4)(new_center - HALF_ONE * new_size, new_center + HALF_ONE * new_size - COORDINATE_OFFSET);

        // adjust new corner locations to be within the image region
#ifdef USE_BLOCKED_FORMAT
        INPUT_TYPE2 img_size;
        size_t img_idx1 = INPUT3_GET_INDEX(0, 1, 0, 0);
        size_t img_idx0 = INPUT3_GET_INDEX(0, 0, 0, 0);
        img_size[0] = im_info[img_idx1];
        img_size[1] = im_info[img_idx0];
#else
        const INPUT_TYPE2 img_size = vload2(0, im_info).s10;
#endif
        new_box = fmax(new_box, ZERO4);

        // recompute new width & height
        const INPUT_TYPE2 new_box_size = new_box.hi - new_box.lo + COORDINATE_OFFSET;

        const int refined_offset = roi_count * class_idx + roi_idx;
        vstore4(new_box, refined_offset, refined_boxes);
        refined_box_areas[refined_offset] = new_box_size.x * new_box_size.y;

#ifdef USE_BLOCKED_FORMAT
        const int scores_offset = INPUT2_GET_INDEX(roi_idx, class_idx, 0, 0);
        refined_scores[refined_offset] = scores[scores_offset];
#else
        refined_scores[refined_offset] = scores[offset];
#endif
    }
}

#endif /* EDDO_STAGE_0_REFINE_BOXES */

#ifdef EDDO_STAGE_1_NMS

inline INPUT_TYPE FUNC(jaccard_overlap)(const __global INPUT_TYPE* refined_boxes,
                                        const __global INPUT_TYPE* refined_box_areas,
                                        size_t idx1,
                                        size_t idx2) {
    INPUT_TYPE4 box1 = vload4(idx1, refined_boxes);
    INPUT_TYPE4 box2 = vload4(idx2, refined_boxes);

    const bool bbox_not_covered = any(isgreater((INPUT_TYPE4)(box1.lo, box2.lo), (INPUT_TYPE4)(box2.hi, box1.hi)));
    if (bbox_not_covered) {
        return ZERO;
    }

    INPUT_TYPE2 intersect_min = max(box1.lo, box2.lo);
    INPUT_TYPE2 intersect_max = min(box1.hi, box2.hi);

    INPUT_TYPE2 intersect_size = intersect_max - intersect_min + COORDINATE_OFFSET;

    if (any(islessequal(intersect_size, ZERO2))) {
        return ZERO;
    }

    INPUT_TYPE intersect_area = intersect_size.x * intersect_size.y;
    INPUT_TYPE bbox1_area = refined_box_areas[idx1];
    INPUT_TYPE bbox2_area = refined_box_areas[idx2];

    return intersect_area / (bbox1_area + bbox2_area - intersect_area);
}

inline void FUNC(nms_cf)(const __global INPUT_TYPE* refined_scores,
                         const __global INPUT_TYPE* refined_boxes,
                         const __global INPUT_TYPE* refined_box_areas,
                         size_t class_idx,
                         size_t roi_count,
                         __global ScoreClassIndex* score_class_index_map,
                         __global uint* detection_count) {
    size_t count = 0;
    for (size_t i = 0; i < roi_count; ++i) {
        if (refined_scores[i] > SCORE_THRESHOLD) {
            score_class_index_map[count] = (ScoreClassIndex){refined_scores[i], class_idx, i};
            count++;
        }
    }

    FUNC_CALL(quickSortIterative)(score_class_index_map, 0, count - 1);

    int detections = 0;
    for (size_t i = 0; i < count; ++i) {
        const size_t idx = score_class_index_map[i].box_idx;

        bool keep = true;
        for (size_t k = 0; k < detections; ++k) {
            const size_t kept_idx = score_class_index_map[k].box_idx;
            INPUT_TYPE overlap = FUNC_CALL(jaccard_overlap)(refined_boxes, refined_box_areas, idx, kept_idx);
            if (overlap > NMS_THRESHOLD) {
                keep = false;
                break;
            }
        }
        if (keep) {
            score_class_index_map[detections] = score_class_index_map[i];
            detections++;
        }
    }

    *detection_count = min(POST_NMS_COUNT, detections);
}

KERNEL(eddo_ref_stage_1)
(const __global INPUT_TYPE* refined_scores,
 const __global INPUT_TYPE* refined_boxes,
 const __global INPUT_TYPE* refined_box_areas,
 __global ScoreClassIndex* score_class_index_map,
 __global uint* detection_count) {
    size_t total_detections_num = 0;

    // FIXME: figure out how to parallelize this!!!
#ifdef CLASS_AGNOSTIC_BOX_REGRESSION
    for (int class_idx = 1; class_idx < NUM_CLASSES; ++class_idx) {
#else
    for (int class_idx = 0; class_idx < NUM_CLASSES; ++class_idx) {
#endif
        FUNC_CALL(nms_cf)
        (&refined_scores[ROI_COUNT * class_idx],
         &refined_boxes[ROI_COUNT * 4 * class_idx],
         &refined_box_areas[ROI_COUNT * class_idx],
         class_idx,
         ROI_COUNT,
         &score_class_index_map[total_detections_num],
         detection_count);
        total_detections_num += *detection_count;
    }

    *detection_count = total_detections_num;
}

#endif /* EDDO_STAGE_1_NMS */

#ifdef EDDO_STAGE_2_TOPK

KERNEL(eddo_ref_stage_2)
(__global ScoreClassIndex* score_class_index_map, const __global uint* detection_count) {
    if (*detection_count > MAX_DETECTIONS_PER_IMAGE) {
        FUNC_CALL(quickSortIterative)(score_class_index_map, 0, *detection_count - 1);
    }
}

#endif /* EDDO_STAGE_2_TOPK */

#ifdef EDDO_STAGE_3_COPY_OUTPUT

KERNEL(eddo_ref_stage_3)
(const __global ScoreClassIndex* score_class_index_map,
 const __global uint* detection_count,
 const __global INPUT_TYPE* refined_boxes,
 __global OUTPUT_TYPE* output_boxes,
 __global OUTPUT1_TYPE* output_classes,
 __global OUTPUT2_TYPE* output_scores) {
    size_t i = get_global_id(0);

#ifdef USE_BLOCKED_FORMAT
    size_t idx0 = OUTPUT_GET_INDEX(i, 0, 0, 0);
    size_t idx1 = OUTPUT_GET_INDEX(i, 1, 0, 0);
    size_t idx2 = OUTPUT_GET_INDEX(i, 2, 0, 0);
    size_t idx3 = OUTPUT_GET_INDEX(i, 3, 0, 0);

    size_t idx_i4 = OUTPUT1_GET_INDEX(i, 0, 0, 0);
    size_t idx_i5 = OUTPUT2_GET_INDEX(i, 0, 0, 0);
#endif
    if (i < *detection_count) {
        OUTPUT_TYPE score = score_class_index_map[i].score;
        OUTPUT1_TYPE cls = score_class_index_map[i].class_idx;
        OUTPUT1_TYPE idx = score_class_index_map[i].box_idx;

#ifdef USE_BLOCKED_FORMAT
        INPUT_TYPE4 res = vload4(ROI_COUNT * cls + idx, refined_boxes);

        output_boxes[idx0] = res[0];
        output_boxes[idx1] = res[1];
        output_boxes[idx2] = res[2];
        output_boxes[idx3] = res[3];
        output_scores[idx_i4] = score;
        output_classes[idx_i5] = cls;
#else
        vstore4(vload4(ROI_COUNT * cls + idx, refined_boxes), i, output_boxes);
        output_scores[i] = score;
        output_classes[i] = cls;
#endif
    } else {

#ifdef USE_BLOCKED_FORMAT
        output_boxes[idx0] = ZERO;
        output_boxes[idx1] = ZERO;
        output_boxes[idx2] = ZERO;
        output_boxes[idx3] = ZERO;
        output_scores[idx_i4] = ZERO;
        output_classes[idx_i5] = 0;
#else
        vstore4(ZERO4, i, output_boxes);
        output_scores[i] = ZERO;
        output_classes[i] = 0;
#endif
    }
}

#endif /* EDDO_STAGE_3_COPY_OUTPUT */

#undef INPUT_TYPE
#undef INPUT_TYPE2
#undef INPUT_TYPE4
#undef HALF_ONE
#undef ZERO
#undef ONE
#undef ZERO2
#undef ZERO4
#undef COORDINATE_OFFSET
#undef DELTA_WEIGHTS
#undef MAX_DELTA_LOG_SIZE
