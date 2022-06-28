// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if INPUT0_TYPE_SIZE == 2 //f16
    #define HALF_ONE 0.5h
    #define ZERO 0.0h

    #ifdef NORMALIZED
      #define COORDINATES_OFFSET  0.0h
    #else
      #define COORDINATES_OFFSET  1.0h
    #endif

#else

    #define HALF_ONE 0.5f
    #define ZERO 0.0f

    #ifdef NORMALIZED
        #define COORDINATES_OFFSET  0.0f
    #else
        #define COORDINATES_OFFSET  1.0f
    #endif

#endif

#ifdef GENERATE_PROPOSALS_STAGE_0

// 0. Refine anchors
KERNEL(generate_proposals_ref_stage_0)
(const __global INPUT0_TYPE* im_info,
 const __global INPUT0_TYPE* anchors,
 const __global INPUT0_TYPE* deltas,
 const __global INPUT0_TYPE* scores,
 __global INPUT0_TYPE* proposals) {
    const INPUT0_TYPE img_H = im_info[0];
    const INPUT0_TYPE img_W = im_info[1];
    const INPUT0_TYPE scale_H = im_info[2];
    const INPUT0_TYPE scale_W = im_info[2]; // may be 3
    const float min_box_H = MIN_SIZE * scale_H;
    const float min_box_W = MIN_SIZE * scale_W;

    const uint h = get_global_id(0);
    const uint w = get_global_id(1);
    const uint bf   = (uint)get_global_id(2);
    const uint anchor = bf % INPUT0_FEATURE_NUM;
    const uint b    = bf / INPUT0_FEATURE_NUM;

    const uint offset = h * BOTTOM_W + w;
    const uint batch_offset = b * ANCHORS_NUM * BOTTOM_AREA;
    const uint anchor_offset = BOTTOM_AREA * anchor;
    const uint proposal_batch_offset = b * NUM_PROPOSALS * 6/*5*/;

    const uint anchor_idx = (offset * ANCHORS_NUM + anchor) * 4;
    const uint score_idx = batch_offset + anchor_offset + offset;
    const uint delta_idx = anchor_offset * 4 + offset;
    const uint proposal_idx = proposal_batch_offset + (offset * ANCHORS_NUM + anchor) * 6/*5*/;


    INPUT0_TYPE x0 = anchors[anchor_idx + 0];
    INPUT0_TYPE y0 = anchors[anchor_idx + 1];
    INPUT0_TYPE x1 = anchors[anchor_idx + 2];
    INPUT0_TYPE y1 = anchors[anchor_idx + 3];

    const INPUT0_TYPE dx = deltas[delta_idx + 0 * BOTTOM_AREA];
    const INPUT0_TYPE dy = deltas[delta_idx + 1 * BOTTOM_AREA];
    const INPUT0_TYPE d_log_w = deltas[delta_idx + 2 * BOTTOM_AREA];
    const INPUT0_TYPE d_log_h = deltas[delta_idx + 3 * BOTTOM_AREA];

    const INPUT0_TYPE score = scores[score_idx];

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

    proposals[proposal_idx + 0] = x0;
    proposals[proposal_idx + 1] = y0;
    proposals[proposal_idx + 2] = x1;
    proposals[proposal_idx + 3] = y1;
    proposals[proposal_idx + 4] = score;
    proposals[proposal_idx + 5] = ((min_box_W <= box_w) && (min_box_H <= box_h)) ? 1 : 0;

    //barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

/*
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    const bool debug = (h==0) && (w==0) && (bf==0);
    if (debug) {
        for(uint bb=0; bb<INPUT0_BATCH_NUM; ++bb) {
            printf("\nGPU refine_anchors result: batch=%d\n", bb);
            for (uint i=0; i<NUM_PROPOSALS; ++i) {
                printf("\ni=%d ", i);
                for (uint j=0; j<6; ++j) {
                    printf("%f ", proposals[i*6+j]);
                }
            }

        }
        printf("\n");
    }
*/
}

#endif /* GENERATE_PROPOSALS_STAGE_0 */

#ifdef GENERATE_PROPOSALS_STAGE_1

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
KERNEL(generate_proposals_ref_stage_1)(__global INPUT0_TYPE* proposals) {
    const uint batch = get_global_id(0);

    __global Box* boxes = (__global Box*)(proposals + batch * NUM_PROPOSALS * 6);

    FUNC_CALL(quickSortIterative)(boxes, 0, NUM_PROPOSALS-1);
    //barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

/*
    const bool debug = (batch==0);
    if (debug) {
        for(uint bb=0; bb<INPUT0_BATCH_NUM; ++bb) {
            printf("\nGPU partial_sort result: batch=%d\n", bb);
            for (uint i=0; i<NUM_PROPOSALS; ++i) {
                printf("\ni=%d ", i);
                for (uint j=0; j<6; ++j) {  // 6 - 5
                    printf("%f ", proposals[bb * NUM_PROPOSALS * 6 + i*6+j]);
                }
            }

        }
        printf("\n");
    }
*/
}
#endif /* GENERATE_PROPOSALS_STAGE_1 */

#ifdef GENERATE_PROPOSALS_STAGE_2

// 2. NMS
KERNEL(generate_proposals_ref_stage_2)
(const __global INPUT0_TYPE* boxes,
       __global size_t* out_indices,
       __global ROI_NUM_TYPE* num_outputs) {

    const bool debug = (get_global_id(0)==0);

    const uint batch = get_global_id(0);
    const uint batch_offset = batch * NUM_PROPOSALS/*PRE_NMS_TOPN*/ * 6;

/*
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
*/

    uint count = 0;
    // common for all the EU or not?
    __local uint index_out[INPUT0_BATCH_NUM * POST_NMS_COUNT]/* = {0}*/;

    __local bool is_dead[INPUT0_BATCH_NUM * PRE_NMS_TOPN]/* = {0}*/;
    for (uint box = 0; box < PRE_NMS_TOPN; ++box) {
        is_dead[batch * PRE_NMS_TOPN + box] = boxes[batch_offset + 6 * box + 5] == 0.0f;
    }
/*

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    if (debug) {
        for(uint bb=0; bb<INPUT0_BATCH_NUM; ++bb) {
            printf("\nGPU is_dead batch=%d\n", bb);
            for (uint i=0; i < PRE_NMS_TOPN; ++i) {
                printf("[%d]=%d\n", i, is_dead[bb * PRE_NMS_TOPN + i]);
            }
        }
        printf("\n");
    }
*/

    for (uint box = 0; box < PRE_NMS_TOPN; ++box) {
        //const uint box = batch * PRE_NMS_TOPN + b;
        if (is_dead[batch * PRE_NMS_TOPN + box])
            continue;

        index_out[batch * POST_NMS_COUNT + count++] = box;
        if (count == POST_NMS_COUNT)
            break;

        const uint box_offset = batch_offset + box * 6/*5*/;
        const INPUT0_TYPE x0i = boxes[box_offset + 0];
        const INPUT0_TYPE y0i = boxes[box_offset + 1];
        const INPUT0_TYPE x1i = boxes[box_offset + 2];
        const INPUT0_TYPE y1i = boxes[box_offset + 3];

        const INPUT0_TYPE a_width = x1i - x0i;
        const INPUT0_TYPE a_height = y1i - y0i;
        const INPUT0_TYPE a_area = (a_width + COORDINATES_OFFSET) * (a_height + COORDINATES_OFFSET);

        for (uint tail = box + 1; tail < PRE_NMS_TOPN; ++tail) {
            const uint tail_offset = batch_offset + tail * 6/*5*/;
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

    num_outputs[batch] = count;

    for (uint i = 0; i < count; ++i) {
        out_indices[batch * POST_NMS_COUNT + i] = index_out[batch * POST_NMS_COUNT + i];
    }

    //barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

/*
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    if (debug) {
        for(uint bb = 0; bb < INPUT0_BATCH_NUM; ++bb) {
            printf("\nGPU nms result: batch=%d, num_outputs=%d\n", bb, num_outputs[bb]);
            for (uint i = 0; i < num_outputs[bb]; ++i) {
                printf("\ni=%d %d", i, out_indices[bb * POST_NMS_COUNT + i]);
            }
        }
        printf("\n");
    }
*/
}
#endif /* GENERATE_PROPOSALS_STAGE_2 */

#ifdef GENERATE_PROPOSALS_STAGE_3

// 3. Convert proposals to rois and roi_scores
KERNEL(generate_proposals_ref_stage_3)
(const __global INPUT0_TYPE* boxes,
 const __global size_t* out_indices,
 const __global ROI_NUM_TYPE* num_outputs,
 __global OUTPUT_TYPE* rois,
 __global OUTPUT_TYPE* roi_scores) {

    const uint batch = get_global_id(0);
    const uint output_index = get_global_id(1);

    uint number_of_previous_outputs = 0;
    for (uint b = 0; b < batch; ++b) {
        number_of_previous_outputs += num_outputs[b];
    }

    const uint score_output_index = number_of_previous_outputs + output_index;
    const uint roi_output_index = score_output_index * 4;

    if (output_index < num_outputs[batch]) {
        const uint box_index = (batch * NUM_PROPOSALS + out_indices[batch * POST_NMS_COUNT + output_index]) * 6;

        rois[roi_output_index + 0] = boxes[box_index + 0];
        rois[roi_output_index + 1] = boxes[box_index + 1];
        rois[roi_output_index + 2] = boxes[box_index + 2];
        rois[roi_output_index + 3] = boxes[box_index + 3];

        roi_scores[score_output_index] = boxes[box_index + 4];
    } else {
        rois[roi_output_index + 0] = 0.0f;
        rois[roi_output_index + 1] = 0.0f;
        rois[roi_output_index + 2] = 0.0f;
        rois[roi_output_index + 3] = 0.0f;

        roi_scores[score_output_index] = 0.0f;
    }

/*
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    const bool debug = get_global_id(0) == 0 && get_global_id(1) == 0;
    if (debug) {
        uint roi_index = 0;
        uint roi_score_index = 0;

        for (uint batch=0; batch < INPUT0_BATCH_NUM; ++batch) {
            printf("GPU batch=%d num_outputs=%d\n", batch, num_outputs[batch]);

            for (uint i=0; i<POST_NMS_COUNT; ++i) {
                printf("%d: %f %f %f %f   %f\n",
                       i, rois[roi_index + 0], rois[roi_index + 1], rois[roi_index + 2], rois[roi_index + 3], roi_scores[roi_score_index]);
                roi_index += 4;
                ++roi_score_index;
            }
        }
    }
*/
}
#endif /* GENERATE_PROPOSALS_STAGE_3 */

#undef ZERO
#undef HALF_ONE
