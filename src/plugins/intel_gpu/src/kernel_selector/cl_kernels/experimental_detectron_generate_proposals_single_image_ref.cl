// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if INPUT0_TYPE_SIZE == 2 //f16
#define HALF_ONE 0.5h
#else
#define HALF_ONE 0.5f
#endif

#define ZERO INPUT0_VAL_ZERO

#ifdef EDGPSI_STAGE_0

#define COORDINATES_OFFSET INPUT0_VAL_ONE

// 0. Refine anchors
KERNEL(edgpsi_ref_stage_0)
(const __global INPUT0_TYPE* im_info,
 const __global INPUT1_TYPE* anchors,
 const __global INPUT2_TYPE* deltas,
 const __global INPUT3_TYPE* scores,
 __global OUTPUT_TYPE* proposals) {
    const INPUT0_TYPE img_H = im_info[INPUT0_GET_INDEX(0, 0, 0, 0)];
    const INPUT0_TYPE img_W = im_info[INPUT0_GET_INDEX(1, 0, 0, 0)];

    const uint h = get_global_id(0);
    const uint w = get_global_id(1);
    const uint anchor = get_global_id(2);

    const uint anchor_idx = (h * BOTTOM_W + w) * ANCHORS_NUM + anchor;

    INPUT0_TYPE x0 = anchors[INPUT1_GET_INDEX(anchor_idx, 0, 0, 0)];
    INPUT0_TYPE y0 = anchors[INPUT1_GET_INDEX(anchor_idx, 1, 0, 0)];
    INPUT0_TYPE x1 = anchors[INPUT1_GET_INDEX(anchor_idx, 2, 0, 0)];
    INPUT0_TYPE y1 = anchors[INPUT1_GET_INDEX(anchor_idx, 3, 0, 0)];

    const INPUT0_TYPE dx = deltas[INPUT2_GET_INDEX(anchor * 4 + 0 , h, w, 0)];
    const INPUT0_TYPE dy = deltas[INPUT2_GET_INDEX(anchor * 4 + 1 , h , w, 0)];
    const INPUT0_TYPE d_log_w = deltas[INPUT2_GET_INDEX(anchor * 4 + 2 , h, w, 0)];
    const INPUT0_TYPE d_log_h = deltas[INPUT2_GET_INDEX(anchor * 4 + 3 , h, w, 0)];

    const INPUT0_TYPE score = scores[INPUT3_GET_INDEX(anchor, h, w, 0)];

    // width & height of box
    const INPUT0_TYPE ww = x1 - x0 + COORDINATES_OFFSET;
    const INPUT0_TYPE hh = y1 - y0 + COORDINATES_OFFSET;
    // center location of box
    const INPUT0_TYPE ctr_x = x0 + HALF_ONE * ww;
    const INPUT0_TYPE ctr_y = y0 + HALF_ONE * hh;

    // new center location according to deltas (dx, dy)
    const INPUT0_TYPE pred_ctr_x = dx * ww + ctr_x;
    const INPUT0_TYPE pred_ctr_y = dy * hh + ctr_y;
    // new width & height according to deltas d(log w), d(log h)
    const INPUT0_TYPE pred_w = exp(min(d_log_w, TO_INPUT0_TYPE(MAX_DELTA_LOG_WH))) * ww;
    const INPUT0_TYPE pred_h = exp(min(d_log_h, TO_INPUT0_TYPE(MAX_DELTA_LOG_WH))) * hh;

    // update upper-left corner location
    x0 = pred_ctr_x - HALF_ONE * pred_w;
    y0 = pred_ctr_y - HALF_ONE * pred_h;
    // update lower-right corner location
    x1 = pred_ctr_x + HALF_ONE * pred_w - COORDINATES_OFFSET;
    y1 = pred_ctr_y + HALF_ONE * pred_h - COORDINATES_OFFSET;

    // adjust new corner locations to be within the image region
    x0 = max(ZERO, min(x0, img_W - COORDINATES_OFFSET));
    y0 = max(ZERO, min(y0, img_H - COORDINATES_OFFSET));
    x1 = max(ZERO, min(x1, img_W - COORDINATES_OFFSET));
    y1 = max(ZERO, min(y1, img_H - COORDINATES_OFFSET));

    // recompute new width & height
    const INPUT0_TYPE box_w = x1 - x0 + COORDINATES_OFFSET;
    const INPUT0_TYPE box_h = y1 - y0 + COORDINATES_OFFSET;

    const uint proposal_idx = anchor_idx * 5;
    proposals[proposal_idx + 0] = x0;
    proposals[proposal_idx + 1] = y0;
    proposals[proposal_idx + 2] = x1;
    proposals[proposal_idx + 3] = y1;
    proposals[proposal_idx + 4] = ((MIN_SIZE <= box_w) && (MIN_SIZE <= box_h)) ? score : 0.f;
}

#undef COORDINATES_OFFSET

#endif /* EDGPSI_STAGE_0 */

#ifdef EDGPSI_STAGE_1
#define Box FUNC(_Box)
typedef struct __attribute__((__packed__)) {
    INPUT0_TYPE x0;
    INPUT0_TYPE y0;
    INPUT0_TYPE x1;
    INPUT0_TYPE y1;
    INPUT0_TYPE score;
} Box;

inline void FUNC(swap_box)(__global Box* a, __global Box* b) {
    const Box temp = *a;
    *a = *b;
    *b = temp;
}

inline int FUNC(partition)(__global Box* arr, int l, int h) {
    static int static_counter = 0;
    static_counter++;
    int pivot_idx = l;
    if (static_counter%3 == 0) { //cyclic pivot selection rotation
        pivot_idx = (l+h)/2;
    }
    if (static_counter%3 == 1) {
        pivot_idx = h;
    }
    INPUT0_TYPE pivotScore = arr[pivot_idx].score;
    FUNC_CALL(swap_box)(&arr[h], &arr[pivot_idx]);
    int i = (l - 1);
    for (int j = l; j <= h - 1; j++) {
        if (arr[j].score > pivotScore) {
            i++;
            FUNC_CALL(swap_box)(&arr[i], &arr[j]);
        }
    }
    FUNC_CALL(swap_box)(&arr[i + 1], &arr[h]);
    return (i + 1);
}

inline void FUNC(bubbleSortIterative)(__global Box* arr, int l, int h) {
    for (int i = 0; i < h - l; i++) {
        bool swapped = false;
        for (int j = l; j < h - i; j++) {
            if ((arr[j].score > arr[j + 1].score)) {
                FUNC_CALL(swap_box)(&arr[j], &arr[j + 1]);
                swapped = true;
            }
        }

        if (!swapped)
            break;
    }
}

inline void FUNC(quickSelectIterative)(__global Box* arr, int l, int h) {
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
        if (p - 1 > l && l < PRE_NMS_TOPN) {
            if (top >= (kStackSize - 1)) {
                FUNC_CALL(bubbleSortIterative)(arr, l, p - 1);
            } else {
                stack[++top] = l;
                stack[++top] = p - 1;
            }
        }

        // If there are elements on right side of pivot,
        // then push right side to stack
        if (p + 1 < h && p + 1 < PRE_NMS_TOPN) {
            if (top >= (kStackSize - 1)) {
                FUNC_CALL(bubbleSortIterative)(arr, p + 1, h);
            } else {
                stack[++top] = p + 1;
                stack[++top] = h;
            }
        }
    }
}

// 1. Sort boxes by scores
KERNEL(edgpsi_ref_stage_1)(__global OUTPUT_TYPE* proposals) {
    __global Box* boxes = (__global Box*)proposals;

    FUNC_CALL(quickSelectIterative)(boxes, 0, NUM_PROPOSALS-1);
}
#undef Box
#endif /* EDGPSI_STAGE_1 */

#ifdef EDGPSI_STAGE_2

// 2. NMS
KERNEL(edgpsi_ref_stage_2)
(const __global INPUT0_TYPE* boxes, __global size_t* out_indices, __global size_t* num_outputs) {
    uint count = 0;
    uint index_out[POST_NMS_COUNT] = {0};

    uint is_dead[PRE_NMS_TOPN] = {0};
    for (uint box = 0; box < PRE_NMS_TOPN; ++box) {
        if (is_dead[box])
            continue;

        index_out[count++] = box;
        if (count == POST_NMS_COUNT)
            break;

        const uint box_offset = box * 5;
        const INPUT0_TYPE x0i = boxes[box_offset + 0];
        const INPUT0_TYPE y0i = boxes[box_offset + 1];
        const INPUT0_TYPE x1i = boxes[box_offset + 2];
        const INPUT0_TYPE y1i = boxes[box_offset + 3];

        const INPUT0_TYPE a_width = x1i - x0i;
        const INPUT0_TYPE a_height = y1i - y0i;
        const INPUT0_TYPE a_area = a_width * a_height;

        for (uint tail = box + 1; tail < PRE_NMS_TOPN; ++tail) {
            const uint tail_offset = tail * 5;
            const INPUT0_TYPE x0j = boxes[tail_offset + 0];
            const INPUT0_TYPE y0j = boxes[tail_offset + 1];
            const INPUT0_TYPE x1j = boxes[tail_offset + 2];
            const INPUT0_TYPE y1j = boxes[tail_offset + 3];

            const INPUT0_TYPE x0 = max(x0i, x0j);
            const INPUT0_TYPE y0 = max(y0i, y0j);
            const INPUT0_TYPE x1 = min(x1i, x1j);
            const INPUT0_TYPE y1 = min(y1i, y1j);

            const INPUT0_TYPE width = x1 - x0;
            const INPUT0_TYPE height = y1 - y0;
            const INPUT0_TYPE area = max(ZERO, width) * max(ZERO, height);

            const INPUT0_TYPE b_width = x1j - x0j;
            const INPUT0_TYPE b_height = y1j - y0j;
            const INPUT0_TYPE b_area = b_width * b_height;

            const INPUT0_TYPE intersection_area = area / (a_area + b_area - area);

            is_dead[tail] =
                (NMS_THRESHOLD < intersection_area) && (x0i <= x1j) && (y0i <= y1j) && (x0j <= x1i) && (y0j <= y1i);
        }
    }

    *num_outputs = count;
    for (uint i = 0; i < count; ++i) {
        out_indices[i] = index_out[i];
    }
}
#endif /* EDGPSI_STAGE_2 */

#ifdef EDGPSI_STAGE_3

// 3. Convert proposals to rois and roi_scores
KERNEL(edgpsi_ref_stage_3)
(const __global INPUT0_TYPE* boxes,
 const __global size_t* out_indices,
 const __global size_t* num_outputs,
 __global OUTPUT_TYPE* rois,
 __global OUTPUT_TYPE* roi_scores) {
    const uint i = get_global_id(0);
    const uint index = out_indices[i];
    const uint box_offset = index * 5;
    const uint rois_offset = i * 4;

    if (i < *num_outputs) {
        rois[OUTPUT_GET_INDEX(i, 0, 0, 0)] = boxes[box_offset + 0];
        rois[OUTPUT_GET_INDEX(i, 1, 0, 0)] = boxes[box_offset + 1];
        rois[OUTPUT_GET_INDEX(i, 2, 0, 0)] = boxes[box_offset + 2];
        rois[OUTPUT_GET_INDEX(i, 3, 0, 0)] = boxes[box_offset + 3];
        roi_scores[OUTPUT1_GET_INDEX(i, 0, 0, 0)] = boxes[box_offset + 4];
    } else {
        rois[OUTPUT_GET_INDEX(i, 0, 0, 0)] = 0.0f;
        rois[OUTPUT_GET_INDEX(i, 1, 0, 0)] = 0.0f;
        rois[OUTPUT_GET_INDEX(i, 2, 0, 0)] = 0.0f;
        rois[OUTPUT_GET_INDEX(i, 3, 0, 0)] = 0.0f;
        roi_scores[OUTPUT1_GET_INDEX(i, 0, 0, 0)] = 0.0f;
    }
}
#endif /* EDGPSI_STAGE_3 */

#undef HALF_ONE
