
#include "include/batch_headers/data_types.cl"

#define SORT_RESULT_CLASSID 0
#define SORT_RESULT_SCORE 1

#define INPUT_INDICES_TYPE int32

#ifndef HAS_ROISNUM

// KERNEL(whats_your_name_again)
//(const __global INPUT0_TYPE* boxes,
//  const __global INPUT0_TYPE* scores,
//  __global OUTPUT_INDICES_TYPE* selected_indices,
//  __global OUTPUT_INDICES_TYPE* selected_num,
//  __global OUTPUT_TYPE* selected_outputs) {
//
//     const  OUTPUT_TYPE selected_outputs_[] = {
//         0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 1.00, 0.95,
//         0.00, 0.00, 1.00, 1.00,  0.00, 0.90,  0.00, 0.00,
//         1.00, 1.00, 1.00, 0.80,  0.00, 10.00, 1.00, 11.00};
//
//     const  OUTPUT_INDICES_TYPE selected_indices_[] = {3, 0, 0, 3};
//
//     int n = 0;
//     for (; n < 4; ++n) {
//         for (int i = 0; i < 6; ++i) {
//             selected_outputs[6 * n + i] = selected_outputs_[6 * n + i];
//         }
//         selected_indices[n] = selected_indices_[n];
//     }
//     *selected_num = 4;
//     for (; n < OUTPUT_DIM; ++n) {
//         for (int i = 0; i < 6; ++i) {
//             selected_outputs[6 * n + i] = 0;
//         }
//         selected_indices[n] = 0;
//     }
//
//     //barrier(CLK_GLOBAL_MEM_FENCE);
//
//     printf("Two 2 inputs\n");
// }

typedef struct __attribute__((__packed__)) {
    INPUT0_TYPE score;
    INPUT0_TYPE xmin;
    INPUT0_TYPE ymin;
    INPUT0_TYPE xmax;
    INPUT0_TYPE ymax;
    OUTPUT_INDICES_TYPE class_idx;
    OUTPUT_INDICES_TYPE batch_idx;
    OUTPUT_INDICES_TYPE index;

} FUNC(BOX_INFO);

#    define BoxInfo FUNC(BOX_INFO)

inline void FUNC(swap_info)(__global BoxInfo* a, __global BoxInfo* b) {
    const BoxInfo temp = *a;
    *a = *b;
    *b = temp;
}

inline int FUNC(partition)(__global BoxInfo* arr, int l, int h, bool sortByScore) {
    const BoxInfo pivot = arr[h];

    int i = (l - 1);
    for (int j = l; j <= h - 1; j++) {
        if (sortByScore) {
            if ((arr[j].score > pivot.score) || (arr[j].score == pivot.score && arr[j].batch_idx < pivot.batch_idx) ||
                (arr[j].score == pivot.score && arr[j].batch_idx == pivot.batch_idx &&
                 arr[j].class_idx < pivot.class_idx) ||
                (arr[j].score == pivot.score && arr[j].batch_idx == pivot.batch_idx &&
                 arr[j].class_idx == pivot.class_idx && arr[j].index < pivot.index)) {
                i++;
                FUNC_CALL(swap_info)(&arr[i], &arr[j]);
            }
        } else {  // sort by class id
            if ((arr[j].class_idx < pivot.class_idx) ||
                (arr[j].class_idx == pivot.class_idx && arr[j].batch_idx < pivot.batch_idx) ||
                (arr[j].class_idx == pivot.class_idx && arr[j].batch_idx == pivot.batch_idx &&
                 arr[j].score > pivot.score) ||
                (arr[j].class_idx == pivot.class_idx && arr[j].batch_idx == pivot.batch_idx &&
                 arr[j].score == pivot.score && arr[j].index < pivot.index)) {
                i++;
                FUNC_CALL(swap_info)(&arr[i], &arr[j]);
            }
        }
    }
    FUNC_CALL(swap_info)(&arr[i + 1], &arr[h]);
    return (i + 1);
}

inline void FUNC(bubbleSortIterative)(__global BoxInfo* arr, int l, int h, bool sortByScore) {
    for (int i = 0; i < h - l; i++) {
        bool swapped = false;
        for (int j = l; j < h - i; j++) {
            if (sortByScore) {
                if ((arr[j].score > arr[j + 1].score) ||
                    (arr[j].score == arr[j + 1].score && arr[j].batch_idx < arr[j + 1].batch_idx) ||
                    (arr[j].score == arr[j + 1].score && arr[j].batch_idx == arr[j + 1].batch_idx &&
                     arr[j].class_idx < arr[j + 1].class_idx) ||
                    (arr[j].score == arr[j + 1].score && arr[j].batch_idx == arr[j + 1].batch_idx &&
                     arr[j].class_idx == arr[j + 1].class_idx && arr[j].index < arr[j + 1].index)) {
                    FUNC_CALL(swap_info)(&arr[j], &arr[j + 1]);
                    swapped = true;
                }
            } else {  // sort by class id
                if ((arr[j].class_idx < arr[j + 1].class_idx) ||
                    (arr[j].class_idx == arr[j + 1].class_idx && arr[j].batch_idx < arr[j + 1].batch_idx) ||
                    (arr[j].class_idx == arr[j + 1].class_idx && arr[j].batch_idx == arr[j + 1].batch_idx &&
                     arr[j].score > arr[j + 1].score) ||
                    (arr[j].class_idx == arr[j + 1].class_idx && arr[j].batch_idx == arr[j + 1].batch_idx &&
                     arr[j].score == arr[j + 1].score && arr[j].index < arr[j + 1].index)) {
                    FUNC_CALL(swap_info)(&arr[j], &arr[j + 1]);
                    swapped = true;
                }
            }
        }

        if (!swapped)
            break;
    }
}

inline void FUNC(quickSortIterative)(__global BoxInfo* arr, int l, int h, bool sortByScore) {
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
        int p = FUNC_CALL(partition)(arr, l, h, sortByScore);

        // If there are elements on left side of pivot,
        // then push left side to stack
        if (p - 1 > l) {
            if (top >= (kStackSize - 1)) {
                FUNC_CALL(bubbleSortIterative)(arr, l, p - 1, sortByScore);
            } else {
                stack[++top] = l;
                stack[++top] = p - 1;
            }
        }

        // If there are elements on right side of pivot,
        // then push right side to stack
        if (p + 1 < h) {
            if (top >= (kStackSize - 1)) {
                FUNC_CALL(bubbleSortIterative)(arr, p + 1, h, sortByScore);
            } else {
                stack[++top] = p + 1;
                stack[++top] = h;
            }
        }
    }
}

inline INPUT0_TYPE FUNC(intersectionOverUnion)(const __global BoxInfo* i, const __global BoxInfo* j) {
    const INPUT0_TYPE norm = !NORMALIZED;

    INPUT0_TYPE areaI = (i->ymax - i->ymin + norm) * (i->xmax - i->xmin + norm);
    INPUT0_TYPE areaJ = (j->ymax - j->ymin + norm) * (j->xmax - j->xmin + norm);

    if (areaI <= 0.0f || areaJ <= 0.0f) { // FIXME macro
        return 0.0f;
    }

    float intersection_ymin = max(i->ymin, j->ymin);
    float intersection_xmin = max(i->xmin, j->xmin);
    float intersection_ymax = min(i->ymax, j->ymax);
    float intersection_xmax = min(i->xmax, j->xmax);

    float intersection_area = max(intersection_ymax - intersection_ymin + norm, 0.0f) *
                              max(intersection_xmax - intersection_xmin + norm, 0.0f);

    return intersection_area / (areaI + areaJ - intersection_area);
}

inline OUTPUT_INDICES_TYPE FUNC(nms)(const __global INPUT0_TYPE* boxes,
                                     const __global INPUT0_TYPE* scores,
                                     OUTPUT_INDICES_TYPE batch_idx,
                                     OUTPUT_INDICES_TYPE class_idx,
                                     __global BoxInfo* box_info) {
    size_t detection_count = 0;

    for (OUTPUT_INDICES_TYPE box_idx = 0; box_idx < NUM_BOXES; box_idx++) {
        if (scores[box_idx] >= SCORE_THRESHOLD) { /* NOTE: ">=" instead of ">" used in PDPD */
            __global BoxInfo* cur_box_info = box_info + detection_count;
            cur_box_info->class_idx = class_idx;
            cur_box_info->batch_idx = batch_idx;
            cur_box_info->index = box_idx;
            cur_box_info->score = scores[box_idx];
            cur_box_info->xmin = boxes[4 * box_idx + 0];
            cur_box_info->ymin = boxes[4 * box_idx + 1];
            cur_box_info->xmax = boxes[4 * box_idx + 2];
            cur_box_info->ymax = boxes[4 * box_idx + 3];

//            printf("  cur_box_info ymax %f\n", cur_box_info->ymax);
//            printf("  class_idx %d\n", class_idx);
//            printf("  box_idx %d\n", box_idx);

            ++detection_count;
        }
    }

    FUNC_CALL(quickSortIterative)(box_info, 0, detection_count - 1, true);

    if (NMS_TOP_K > -1)
        detection_count = min((size_t)NMS_TOP_K, detection_count);

    printf("detection count %d\n", detection_count);

    INPUT0_TYPE adaptive_threshold = IOU_THRESHOLD;

    size_t selected_size = 0;
    for (size_t i = 0; i < detection_count; ++i) {
        __global BoxInfo* next_candidate = box_info + i;

//        printf("next_candidate.box: %f %f %f %f\n", next_candidate->xmin, next_candidate->ymin, next_candidate->xmax, next_candidate->ymax);
//        printf("  score %f class_idx %d batch_idx %d index %d\n", next_candidate->score, next_candidate->class_idx, next_candidate->batch_idx, next_candidate->index);
        bool should_hard_suppress = false;

        if (NMS_ETA < 1 && adaptive_threshold > 0.5) // FIXME: macro for half
            adaptive_threshold *= NMS_ETA;

        for (size_t j = 0; j < selected_size; ++j) {
            __global BoxInfo* selected = box_info + j;
            float iou = FUNC_CALL(intersectionOverUnion)(box_info + i, box_info + j);

//            printf("next_candidate.box: %f %f %f %f\n", next_candidate->xmin, next_candidate->ymin, next_candidate->xmax, next_candidate->ymax);
//            printf("selected.box: %f %f %f %f\n", selected->xmin, selected->ymin, selected->xmax, selected->ymax);
//            printf("  class_idx: %d, i: %d, j: %d, iou: %f\n", class_idx, i, j, iou);
            if (iou >= adaptive_threshold) {

                should_hard_suppress = true;
            }
        }
        if (!should_hard_suppress) {
            box_info[selected_size] = box_info[i];
            ++selected_size;
        }
    }


    //printf("batch_idx %d class_idx %d detection count: %d\n", (int)batch_idx, (int)class_idx, (int)selected_size);

    return selected_size;
}

inline OUTPUT_INDICES_TYPE FUNC(multiclass_nms)(const __global INPUT0_TYPE* boxes,
                                                const __global INPUT0_TYPE* scores,
                                                OUTPUT_INDICES_TYPE batch_idx,
                                                __global BoxInfo* box_info) {
    OUTPUT_INDICES_TYPE detection_count = 0;
    for (OUTPUT_INDICES_TYPE class_idx = 0; class_idx < NUM_CLASSES; ++class_idx) {
        if (class_idx == BACKGROUND_CLASS)
            continue;

        detection_count +=
            FUNC_CALL(nms)(boxes, scores + class_idx * NUM_BOXES, batch_idx, class_idx, box_info + detection_count);
        //printf("again dc %d\n", (int)detection_count);
    }

    FUNC_CALL(quickSortIterative)(box_info, 0, detection_count - 1, true);

    if (KEEP_TOP_K > -1)
        detection_count = min(detection_count, KEEP_TOP_K);

#if !(SORT_RESULT_ACROSS_BATCH) && (SORT_RESULT_TYPE == SORT_RESULT_CLASSID)
    printf("Oops\n");
    FUNC_CALL(quickSortIterative)(box_info, 0, detection_count - 1, false);
#endif

    //printf("batch_idx %d detection count: %d\n", (int)batch_idx, (int)detection_count);
    return detection_count;
}

KERNEL(whats_your_name_again)
(const __global INPUT0_TYPE* boxes,
 const __global INPUT0_TYPE* scores,
 __global OUTPUT_INDICES_TYPE* selected_indices,
 __global OUTPUT_INDICES_TYPE* selected_num,
 __global BoxInfo* box_info,
 __global OUTPUT_TYPE* selected_outputs) {
    OUTPUT_INDICES_TYPE offset = 0;
    for (OUTPUT_INDICES_TYPE i = 0; i < NUM_BATCHES; ++i) {
        const __global INPUT0_TYPE* boxesPtr = boxes + i * NUM_BOXES * 4;
        const __global INPUT0_TYPE* scoresPtr = scores + i * NUM_CLASSES * NUM_BOXES;

        offset += (i == 0 ? 0 : selected_num[i - 1]);
        //printf("offset: %d\n", offset);
        OUTPUT_INDICES_TYPE nselected = FUNC_CALL(multiclass_nms)(boxesPtr, scoresPtr, i, box_info + offset);
        selected_num[i] = nselected;
    }
    offset += selected_num[NUM_BATCHES - 1];

#    if SORT_RESULT_ACROSS_BATCH
    printf(">>> sort across batch!!!\n");
#        if SORT_RESULT_TYPE == SORT_RESULT_SCORE
    printf(">>> sort across batch, by score\n");
    FUNC_CALL(quickSortIterative)(box_info, 0, offset - 1, true);
#        elif SORT_RESULT_TYPE == SORT_RESULT_CLASSID
    printf(">>> sort across batch, by class\n");
    FUNC_CALL(quickSortIterative)(box_info, 0, offset - 1, false);
#        endif
#    endif  // SORT_RESULT_ACROSS_BATCH

    //size_t output_size = min(selected_num[NUM_BATCHES - 1], OUTPUT_DIM);
    size_t output_size = offset;
    size_t idx;
    for (idx = 0; idx < output_size; ++idx) {
        const __global BoxInfo* info = box_info + idx;
        selected_outputs[6 * idx + 0] = (INPUT0_TYPE)info->class_idx;
        selected_outputs[6 * idx + 1] = info->score;
        selected_outputs[6 * idx + 2] = info->xmin;
        selected_outputs[6 * idx + 3] = info->ymin;
        selected_outputs[6 * idx + 4] = info->xmax;
        selected_outputs[6 * idx + 5] = info->ymax;

        selected_indices[idx] = info->batch_idx * NUM_BOXES + info->index;
    }

    // TODO: filler

    printf("Two 2 inputs\n");
}

#else  // HAS_ROISNUNM

#    define INPUT_INDICES_TYPE INPUT2_TYPE

KERNEL(whats_your_name_again)
(const __global INPUT0_TYPE* boxes,
 const __global INPUT0_TYPE* scores,
 const __global INPUT_INDICES_TYPE* roisnum,
 __global OUTPUT_INDICES_TYPE* selected_indices,
 __global OUTPUT_INDICES_TYPE* selected_num,
 __global OUTPUT_TYPE* selected_outputs) {
    printf("Three 3 inputs\n");
}

#endif  // HAS_ROISNUM