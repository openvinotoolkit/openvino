// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if INPUT0_TYPE_SIZE == 2 //f16
    #define HALF_ONE 0.5h
#else
    #define HALF_ONE 0.5f
#endif

#define ZERO INPUT0_VAL_ZERO

#ifdef NORMALIZED
    #define COORDINATES_OFFSET INPUT0_VAL_ZERO
#else
    #define COORDINATES_OFFSET INPUT0_VAL_ONE
#endif

#ifdef GENERATE_PROPOSALS_STAGE_0

// 0. Refine anchors
KERNEL(generate_proposals_ref_stage_0)
(const __global INPUT0_TYPE* im_info,
 const __global INPUT1_TYPE* anchors,
 const __global INPUT2_TYPE* deltas,
 const __global INPUT3_TYPE* scores,
 __global OUTPUT_TYPE* proposals) {
    const uint h = get_global_id(0);
    const uint w = get_global_id(1);
    const uint ba = (uint)get_global_id(2);
    const uint batch = ba / INPUT0_FEATURE_NUM;
    const uint anchor = ba % INPUT0_FEATURE_NUM;

    const INPUT0_TYPE img_H = im_info[INPUT0_GET_INDEX(batch, 0, 0, 0)];
    const INPUT0_TYPE img_W = im_info[INPUT0_GET_INDEX(batch, 1, 0, 0)];
    const INPUT0_TYPE scale_H = im_info[INPUT0_GET_INDEX(batch, 2, 0, 0)];
    const INPUT0_TYPE scale_W = im_info[INPUT0_GET_INDEX(batch, SCALE_W_INDEX, 0, 0)];
    const float min_box_H = MIN_SIZE * scale_H;
    const float min_box_W = MIN_SIZE * scale_W;

    INPUT0_TYPE x0 = anchors[INPUT1_GET_INDEX(h, w, anchor, 0)];
    INPUT0_TYPE y0 = anchors[INPUT1_GET_INDEX(h, w, anchor, 1)];
    INPUT0_TYPE x1 = anchors[INPUT1_GET_INDEX(h, w, anchor, 2)];
    INPUT0_TYPE y1 = anchors[INPUT1_GET_INDEX(h, w, anchor, 3)];

    const INPUT0_TYPE dx = deltas[INPUT2_GET_INDEX(batch, anchor * 4 + 0 , h, w)];
    const INPUT0_TYPE dy = deltas[INPUT2_GET_INDEX(batch, anchor * 4 + 1 , h , w)];
    const INPUT0_TYPE d_log_w = deltas[INPUT2_GET_INDEX(batch, anchor * 4 + 2 , h, w)];
    const INPUT0_TYPE d_log_h = deltas[INPUT2_GET_INDEX(batch, anchor * 4 + 3 , h, w)];

    const INPUT0_TYPE score = scores[INPUT3_GET_INDEX(batch, anchor, h, w)];

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

    const uint batch_offset = batch * NUM_PROPOSALS * PROPOSAL_SIZE;
    const uint offset = h * BOTTOM_W + w;
    const uint proposal_idx = batch_offset + (offset * ANCHORS_NUM + anchor) * PROPOSAL_SIZE;

    proposals[proposal_idx + 0] = x0;
    proposals[proposal_idx + 1] = y0;
    proposals[proposal_idx + 2] = x1;
    proposals[proposal_idx + 3] = y1;
    proposals[proposal_idx + 4] = score;
    proposals[proposal_idx + 5] = ((min_box_W <= box_w) && (min_box_H <= box_h)) ? 1 : 0;
}

#endif /* GENERATE_PROPOSALS_STAGE_0 */

#ifdef GENERATE_PROPOSALS_STAGE_1
#define Box FUNC(__Box)
typedef struct __attribute__((__packed__)) {
    INPUT0_TYPE x0;
    INPUT0_TYPE y0;
    INPUT0_TYPE x1;
    INPUT0_TYPE y1;
    INPUT0_TYPE score;
    INPUT0_TYPE keep;
} Box;

inline void FUNC(swap_box)(__global Box* a, __global Box* b) {
    const Box temp = *a;
    *a = *b;
    *b = temp;
}

inline int FUNC(partition)(__global Box* arr, int l, int h) {
    INPUT0_TYPE pivotScore = arr[h].score;
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

inline void FUNC(quickSortIterative)(__global Box* arr, int l, int h) {
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

// 1. Sort boxes by scores
KERNEL(generate_proposals_ref_stage_1)(__global OUTPUT_TYPE* proposals) {
    const uint batch = get_global_id(0);

    __global Box* boxes = (__global Box*)(proposals + batch * NUM_PROPOSALS * PROPOSAL_SIZE);

    FUNC_CALL(quickSortIterative)(boxes, 0, NUM_PROPOSALS-1);
}
#undef Box
#endif /* GENERATE_PROPOSALS_STAGE_1 */

#ifdef GENERATE_PROPOSALS_STAGE_2

// 2. NMS
KERNEL(generate_proposals_ref_stage_2)
(const __global INPUT0_TYPE* boxes,
       __global size_t* out_indices,
       __global ROI_NUM_TYPE* num_outputs) {
    const uint batch = get_global_id(0);
    const uint batch_offset = batch * NUM_PROPOSALS * PROPOSAL_SIZE;

    uint count = 0;
    __local uint index_out[INPUT0_BATCH_NUM * POST_NMS_COUNT];
    __local bool is_dead[INPUT0_BATCH_NUM * PRE_NMS_TOPN];

    for (uint box = 0; box < PRE_NMS_TOPN; ++box) {
        is_dead[batch * PRE_NMS_TOPN + box] = boxes[batch_offset + PROPOSAL_SIZE * box + 5] == 0.0f;
    }

    for (uint box = 0; box < PRE_NMS_TOPN; ++box) {
        if (is_dead[batch * PRE_NMS_TOPN + box])
            continue;

        index_out[batch * POST_NMS_COUNT + count++] = box;
        if (count == POST_NMS_COUNT)
            break;

        const uint box_offset = batch_offset + box * PROPOSAL_SIZE;
        const INPUT0_TYPE x0i = boxes[box_offset + 0];
        const INPUT0_TYPE y0i = boxes[box_offset + 1];
        const INPUT0_TYPE x1i = boxes[box_offset + 2];
        const INPUT0_TYPE y1i = boxes[box_offset + 3];

        const INPUT0_TYPE a_width = x1i - x0i;
        const INPUT0_TYPE a_height = y1i - y0i;
        const INPUT0_TYPE a_area = (a_width + COORDINATES_OFFSET) * (a_height + COORDINATES_OFFSET);

        for (uint tail = box + 1; tail < PRE_NMS_TOPN; ++tail) {
            const uint tail_offset = batch_offset + tail * PROPOSAL_SIZE;
            const INPUT0_TYPE x0j = boxes[tail_offset + 0];
            const INPUT0_TYPE y0j = boxes[tail_offset + 1];
            const INPUT0_TYPE x1j = boxes[tail_offset + 2];
            const INPUT0_TYPE y1j = boxes[tail_offset + 3];

            const INPUT0_TYPE x0 = max(x0i, x0j);
            const INPUT0_TYPE y0 = max(y0i, y0j);
            const INPUT0_TYPE x1 = min(x1i, x1j);
            const INPUT0_TYPE y1 = min(y1i, y1j);

            const INPUT0_TYPE width = x1 - x0 + COORDINATES_OFFSET;
            const INPUT0_TYPE height = y1 - y0 + COORDINATES_OFFSET;
            const INPUT0_TYPE area = max(ZERO, width) * max(ZERO, height);

            const INPUT0_TYPE b_width = x1j - x0j;
            const INPUT0_TYPE b_height = y1j - y0j;
            const INPUT0_TYPE b_area = (b_width + COORDINATES_OFFSET) * (b_height + COORDINATES_OFFSET);

            const INPUT0_TYPE intersection_area = area / (a_area + b_area - area);

            if ( (NMS_THRESHOLD < intersection_area) && (x0i <= x1j) && (y0i <= y1j) && (x0j <= x1i) && (y0j <= y1i) ) {
                is_dead[batch * PRE_NMS_TOPN + tail] = true;
            }
        }
    }

    num_outputs[OUTPUT2_GET_INDEX(batch, 0, 0, 0)] = count;

    for (uint i = 0; i < count; ++i) {
        out_indices[batch * POST_NMS_COUNT + i] = index_out[batch * POST_NMS_COUNT + i];
    }
}
#endif /* GENERATE_PROPOSALS_STAGE_2 */

#ifdef GENERATE_PROPOSALS_STAGE_3

// 3. Convert proposals to rois and roi_scores
KERNEL(generate_proposals_ref_stage_3)
(const __global INPUT0_TYPE* boxes,
 const __global size_t* out_indices,
 const __global ROI_NUM_TYPE* num_outputs,
 __global OUTPUT_TYPE* rois,
 __global OUTPUT1_TYPE* roi_scores) {

    uint roi_index = 0;
    for (uint batch = 0; batch < INPUT0_BATCH_NUM; ++batch) {
        for (uint i = 0; i < num_outputs[OUTPUT2_GET_INDEX(batch, 0, 0, 0)]; ++i) {
            const uint box_index = (batch * NUM_PROPOSALS + out_indices[batch * POST_NMS_COUNT + i]) * PROPOSAL_SIZE;

            rois[OUTPUT_GET_INDEX(roi_index, 0, 0, 0)] = boxes[box_index + 0];
            rois[OUTPUT_GET_INDEX(roi_index, 1, 0, 0)] = boxes[box_index + 1];
            rois[OUTPUT_GET_INDEX(roi_index, 2, 0, 0)] = boxes[box_index + 2];
            rois[OUTPUT_GET_INDEX(roi_index, 3, 0, 0)] = boxes[box_index + 3];
            roi_scores[OUTPUT1_GET_INDEX(roi_index, 0, 0, 0)] = boxes[box_index + 4];
            ++roi_index;
        }
    }

    // fill the rest of outputs with zeros
    while(roi_index < INPUT0_BATCH_NUM * POST_NMS_COUNT) {
        rois[OUTPUT_GET_INDEX(roi_index, 0, 0, 0)] = 0.0f;
        rois[OUTPUT_GET_INDEX(roi_index, 1, 0, 0)] = 0.0f;
        rois[OUTPUT_GET_INDEX(roi_index, 2, 0, 0)] = 0.0f;
        rois[OUTPUT_GET_INDEX(roi_index, 3, 0, 0)] = 0.0f;

        roi_scores[roi_index] = 0.0f;
        ++roi_index;
    }
}
#endif /* GENERATE_PROPOSALS_STAGE_3 */

#undef HALF_ONE
#undef ZERO
#undef COORDINATES_OFFSET
