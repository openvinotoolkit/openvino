// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (c) Facebook, Inc. and its affiliates.
// The implementation for rotated boxes intersection is based on the code from:
// https://github.com/facebookresearch/detectron2/blob/v0.6/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_utils.h

#include "include/batch_headers/fetch_data.cl"

/// Kernels
/// 0: Only boxes exceeding SCORE_THRESHOLD are copied to intermediate-buffer0.
///    Set copied box number to intermediate-buffer2
/// 1: Sort the boxes in buffer0 by class.
/// 2: Remove boxes what are over IOU_THRESHOLD.
/// 3: Copy the boxes to output. If SORT_RESULT_DESCENDING is 1, all boxes will be sorted without class distinction.

/// KERNEL_ARGs
///
/// boxes
///  - shape: {num_batches, num_boxes, 4}
/// scores
///  - shape: {num_batches, num_classes, num_boxes}
/// buffer0 (intermediate buffer)
///  - size: batch_num * class_num * boxes_num * sizeof(SBOX_INFO)
///  - desc: filtered and sorted SBOX_INFO list
/// buffer1 (intermediate buffer)
///  - size: batch_num * class_num * boxes_num * sizeof(BOX_INFO)
///  - desc: selected SBOX_INFO list by iou calucation
/// buffer2 (intermediate buffer)
///  - size: batch_num * class_num * sizeof(int)
///  - desc: sorted box num for batch*class

/// optional input variables
/// NUM_SELECT_PER_CLASS_VAL TO_UNIT_TYPE(num_select_per_class[0]),   default is 0
/// IOU_THRESHOLD_VAL        TO_ACCUMULATOR_TYPE(iou_threshold[0]),   default is ACCUMULATOR_VAL_ZERO
/// SCORE_THRESHOLD_VAL      TO_ACCUMULATOR_TYPE(score_threshold[0]), default is ACCUMULATOR_VAL_ZERO
/// SOFT_NMS_SIGMA_VAL       TO_ACCUMULATOR_TYPE(soft_nms_sigma[0]),  default is ACCUMULATOR_VAL_ZERO
/// OUTPUT_NUM               Number of outputs. [OUTPUT_NUM, 3, 1, 1]
/// BUFFER_STRIDE            sizeof(SBOX_INFO) * NUM_BOXES

#define NUM_BATCHES     INPUT0_BATCH_NUM
#define NUM_BOXES       INPUT0_FEATURE_NUM
#define NUM_CLASSES     INPUT1_FEATURE_NUM

typedef struct {
    ushort boxId;
    int suppress_begin_index;
    INPUT1_TYPE score;
} FUNC(SortedBoxInfo);

typedef struct {
    short batchId;
    ushort classId;
    ushort boxId;
    INPUT1_TYPE score;
} FUNC(BoxInfo);

#define SBOX_INFO FUNC(SortedBoxInfo)
#define BOX_INFO FUNC(BoxInfo)

inline COORD_TYPE_4 FUNC(getBoxCoords)(const __global INPUT0_TYPE *boxes, const short batch, const ushort boxId)
{
    COORD_TYPE_4 coords = (COORD_TYPE_4)(boxes[INPUT0_GET_INDEX(batch, boxId, 0, 0)],
                                       boxes[INPUT0_GET_INDEX(batch, boxId, 1, 0)],
                                       boxes[INPUT0_GET_INDEX(batch, boxId, 2, 0)],
                                       boxes[INPUT0_GET_INDEX(batch, boxId, 3, 0)]);

#if !defined(ROTATION) && BOX_ENCODING == 0
    const COORD_TYPE ax1 = min(coords[1], coords[3]);
    const COORD_TYPE ax2 = max(coords[1], coords[3]);
    const COORD_TYPE ay1 = min(coords[0], coords[2]);
    const COORD_TYPE ay2 = max(coords[0], coords[2]);
    coords[1] = ax1;
    coords[3] = ax2;
    coords[0] = ay1;
    coords[2] = ay2;
#endif

    return coords;
}

#ifdef ROTATION

typedef struct {
    float x, y;
} FUNC(Point2D);
#define POINT_2D FUNC(Point2D)

inline void FUNC(getRotatedVertices)(const COORD_TYPE_4 box, const INPUT0_TYPE angle, POINT_2D* pts) {
    const float theta = angle
                        #if ROTATION == 2
                            * -1.0f
                        #endif
                        ;
    float cosTheta2 = cos(theta) * 0.5f;
    float sinTheta2 = sin(theta) * 0.5f;

    // y: top --> down; x: left --> right
    // Left-Down
    pts[0].x = box[0]/*.x_ctr*/ - sinTheta2 * box[3]/*.h*/ - cosTheta2 * box[2]/*.w*/;
    pts[0].y = box[1]/*.y_ctr*/ + cosTheta2 * box[3]/*.h*/ - sinTheta2 * box[2]/*.w*/;
    // Left-Top
    pts[1].x = box[0]/*.x_ctr*/ + sinTheta2 * box[3]/*.h*/ - cosTheta2 * box[2]/*.w*/;
    pts[1].y = box[1]/*.y_ctr*/ - cosTheta2 * box[3]/*.h*/ - sinTheta2 * box[2]/*.w*/;
    // Right-Top
    pts[2].x = 2 * box[0]/*.x_ctr*/ - pts[0].x;
    pts[2].y = 2 * box[1]/*.y_ctr*/ - pts[0].y;
    // Right-Down
    pts[3].x = 2 * box[0]/*.x_ctr*/ - pts[1].x;
    pts[3].y = 2 * box[1]/*.y_ctr*/ - pts[1].y;
}

inline float FUNC(dot2D)(const POINT_2D A, const POINT_2D B) {
    return A.x * B.x + A.y * B.y;
}

inline float FUNC(cross2D)(const POINT_2D A, const POINT_2D B) {
    return A.x * B.y - B.x * A.y;
}

inline int FUNC(getIntersectionPoints)(const POINT_2D* pts1, const POINT_2D* pts2, POINT_2D* intersections) {
    // Line vector
    // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
    POINT_2D vec1[4], vec2[4];
    for (int i = 0; i < 4; i++) {
        vec1[i].x = pts1[(i + 1) % 4].x - pts1[i].x;
        vec1[i].y = pts1[(i + 1) % 4].y - pts1[i].y;
        vec2[i].x = pts2[(i + 1) % 4].x - pts2[i].x;
        vec2[i].y = pts2[(i + 1) % 4].y - pts2[i].y;
    }

    // Line test - test all line combos for intersection
    int num = 0;  // number of intersections
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            // Solve for 2x2 Ax=b
            float det = FUNC_CALL(cross2D)(vec2[j], vec1[i]);
            // This takes care of parallel lines
            if (fabs(det) <= 1e-14f) {
                continue;
            }

            POINT_2D vec12;
            vec12.x= pts2[j].x - pts1[i].x;
            vec12.y= pts2[j].y - pts1[i].y;

            float t1 = FUNC_CALL(cross2D)(vec2[j], vec12) / det;
            float t2 = FUNC_CALL(cross2D)(vec1[i], vec12) / det;

            if (t1 >= 0.0f && t1 <= 1.0f && t2 >= 0.0f && t2 <= 1.0f) {
                intersections[num].x = pts1[i].x + vec1[i].x * t1;
                intersections[num].y = pts1[i].y + vec1[i].y * t1;
                ++num;
            }
        }
    }

    // Check for vertices of rect1 inside rect2
    {
        const POINT_2D AB = vec2[0];
        const POINT_2D DA = vec2[3];
        float ABdotAB = FUNC_CALL(dot2D)(AB, AB);
        float ADdotAD = FUNC_CALL(dot2D)(DA, DA);
        for (int i = 0; i < 4; i++) {
            // assume ABCD is the rectangle, and P is the point to be judged
            // P is inside ABCD iff. P's projection on AB lies within AB
            // and P's projection on AD lies within AD

            POINT_2D AP;
            AP.x = pts1[i].x - pts2[0].x;
            AP.y = pts1[i].y - pts2[0].y;

            float APdotAB = FUNC_CALL(dot2D)(AP, AB);
            float APdotAD = -FUNC_CALL(dot2D)(AP, DA);

            if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) && (APdotAD <= ADdotAD)) {
                intersections[num].x = pts1[i].x;
                intersections[num].y = pts1[i].y;
                ++num;
            }
        }
    }

    // Reverse the check - check for vertices of rect2 inside rect1
    {
        const POINT_2D AB = vec1[0];
        const POINT_2D DA = vec1[3];
        float ABdotAB = FUNC_CALL(dot2D)(AB, AB);
        float ADdotAD = FUNC_CALL(dot2D)(DA, DA);
        for (int i = 0; i < 4; i++) {
            POINT_2D AP;
            AP.x = pts2[i].x - pts1[0].x;
            AP.y = pts2[i].y - pts1[0].y;

            float APdotAB = FUNC_CALL(dot2D)(AP, AB);
            float APdotAD = -FUNC_CALL(dot2D)(AP, DA);

            if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) && (APdotAD <= ADdotAD)) {
                intersections[num].x = pts2[i].x;
                intersections[num].y = pts2[i].y;
                ++num;
            }
        }
    }

    return num;
}

inline void FUNC(swapPoints)(POINT_2D* a, POINT_2D* b)
{
    POINT_2D temp = *a;
    *a = *b;
    *b = temp;
}

inline void FUNC(sortPoints)(POINT_2D* arr, int l, int h)
{
    for (int i = 0; i < h-l; i++) {
        bool swapped = false;

        for (int j = l; j < h-i; j++) {
            bool is_less = false;
            const float temp = FUNC_CALL(cross2D)(arr[j], arr[j+1]);
            if (fabs(temp) < 1e-6f) {
                is_less = FUNC_CALL(dot2D)(arr[j], arr[j]) < FUNC_CALL(dot2D)(arr[j+1], arr[j+1]);
            } else {
                is_less = temp > 0;
            }

            if (is_less) {
                continue;
            }

            FUNC_CALL(swapPoints)(&arr[j], &arr[j+1]);
            swapped = true;
        }

        if (!swapped) {
            break;
        }
    }
}

inline int FUNC(convex_hull_graham)(const POINT_2D* p, const int num_in, POINT_2D* q, bool shift_to_zero) {
    if (num_in < 2) {
        return -1;
    }

    // Step 1:
    // Find point with minimum y
    // if more than 1 points have the same minimum y,
    // pick the one with the minimum x.
    int t = 0;
    for (int i = 1; i < num_in; i++) {
        if (p[i].y < p[t].y || (p[i].y == p[t].y && p[i].x < p[t].x)) {
            t = i;
        }
    }
    const POINT_2D start = p[t];  // starting point

    // Step 2:
    // Subtract starting point from every points (for sorting in the next step)
    for (int i = 0; i < num_in; i++) {
        q[i].x = p[i].x - start.x;
        q[i].y = p[i].y - start.y;
    }

    // Swap the starting point to position 0
    FUNC_CALL(swapPoints)(&q[t], &q[0]);

    // Step 3:
    // Sort point 1 ~ num_in according to their relative cross-product values
    // (essentially sorting according to angles)
    // If the angles are the same, sort according to their distance to origin
    float dist[24];
    for (int i = 0; i < num_in; i++) {
        dist[i] = FUNC_CALL(dot2D)(q[i], q[i]);
    }

    FUNC_CALL(sortPoints)(q, 1, num_in - 1);

    // compute distance to origin after sort, since the points are now different.
    for (int i = 0; i < num_in; i++) {
        dist[i] = FUNC_CALL(dot2D)(q[i], q[i]);
    }

    // Step 4:
    // Make sure there are at least 2 points (that don't overlap with each other)
    // in the stack
    int k;  // index of the non-overlapped second point
    for (k = 1; k < num_in; k++) {
        if (dist[k] > 1e-8f) {
            break;
        }
    }
    if (k == num_in) {
        // We reach the end, which means the convex hull is just one point
        q[0].x = p[t].x;
        q[0].y = p[t].y;
        return 1;
    }

    q[1].x = q[k].x;
    q[1].y = q[k].y;
    int m = 2;  // 2 points in the stack
    // Step 5:
    // Finally we can start the scanning process.
    // When a non-convex relationship between the 3 points is found
    // (either concave shape or duplicated points),
    // we pop the previous point from the stack
    // until the 3-point relationship is convex again, or
    // until the stack only contains two points
    for (int i = k + 1; i < num_in; i++) {
        POINT_2D diff1, diff2;
        diff1.x = q[i].x - q[m - 2].x;
        diff1.y = q[i].y - q[m - 2].y;
        diff2.x = q[m - 1].x - q[m - 2].x;
        diff2.y = q[m - 1].y - q[m - 2].y;

        float cross2d_diff = FUNC_CALL(cross2D)(diff1, diff2);

        while (m > 1 && cross2d_diff >= 0) {
            m--;
        }
        q[m].x = q[i].x;
        q[m].y = q[i].y;
        ++m;
    }

    // Step 6 (Optional):
    // In general sense we need the original coordinates, so we
    // need to shift the points back (reverting Step 2)
    // But if we're only interested in getting the area/perimeter of the shape
    // We can simply return.
    if (!shift_to_zero) {
        for (int i = 0; i < m; i++) {
            q[i].x += start.x;
            q[i].y += start.y;
        }
    }

    return m;
}

inline float FUNC(polygon_area)(const POINT_2D* q, const int m) {
    if (m <= 2) {
        return 0.f;
    }

    float area = 0.f;
    for (int i = 1; i < m - 1; i++) {
        POINT_2D diff1, diff2;
        diff1.x = q[i].x - q[0].x;
        diff1.y = q[i].y - q[0].y;
        diff2.x = q[i + 1].x - q[0].x;
        diff2.y = q[i + 1].y - q[0].y;
        float cross_result = FUNC_CALL(cross2D)(diff1, diff2);

        area += fabs(cross_result);
    }

    return area / 2.0f;
}

inline float FUNC(rotatedBoxesIntersection)(const COORD_TYPE_4 boxA, const INPUT0_TYPE angleA,
        const COORD_TYPE_4 boxB, const INPUT0_TYPE angleB) {
    // There are up to 4 x 4 + 4 + 4 = 24 intersections (including dups) returned
    // from get_intersection_points
    POINT_2D intersectPts[24], orderedPts[24];
    POINT_2D pts1[4];
    POINT_2D pts2[4];
    FUNC_CALL(getRotatedVertices)(boxA, angleA, pts1);
    FUNC_CALL(getRotatedVertices)(boxB, angleB, pts2);
    // Find points defining area of the boxes intersection
    int num = FUNC_CALL(getIntersectionPoints)(pts1, pts2, intersectPts);

    if (num <= 2) {
        return 0.f;
    }

    // Convex Hull to order the intersection points in clockwise order and find
    // the contour area.
    int num_convex = FUNC_CALL(convex_hull_graham)(intersectPts, num, orderedPts, true);
    return FUNC_CALL(polygon_area)(orderedPts, num_convex);
}


inline float FUNC(intersectionOverUnion)(const COORD_TYPE_4 boxA, const INPUT0_TYPE angleA,
        const COORD_TYPE_4 boxB, const INPUT0_TYPE angleB)
{
    const float areaA = convert_float(boxA[3]) * convert_float(boxA[2]);
    const float areaB = convert_float(boxB[3]) * convert_float(boxB[2]);

    if (areaA <= 0.0f || areaB <= 0.0f)
        return 0.0f;

    const float intersection_area = FUNC_CALL(rotatedBoxesIntersection)(boxA, angleA, boxB, angleB);
    const float union_area = areaA + areaB - intersection_area;
    return intersection_area / union_area;
}

#else

inline float FUNC(intersectionOverUnion)(const COORD_TYPE_4 boxA, const COORD_TYPE_4 boxB)
{
#if !defined(ROTATION) && BOX_ENCODING == 0
    /// CORNER
    const float areaA = convert_float(boxA[3] - boxA[1]) * convert_float(boxA[2] - boxA[0]);
    const float areaB = convert_float(boxB[3] - boxB[1]) * convert_float(boxB[2] - boxB[0]);

    const COORD_TYPE intersection_ymin = max(boxA[0], boxB[0]);
    const COORD_TYPE intersection_xmin = max(boxA[1], boxB[1]);
    const COORD_TYPE intersection_ymax = min(boxA[2], boxB[2]);
    const COORD_TYPE intersection_xmax = min(boxA[3], boxB[3]);
#else
    /// CENTER
    const float areaA = convert_float(boxA[3]) * convert_float(boxA[2]);
    const float areaB = convert_float(boxB[3]) * convert_float(boxB[2]);
    const COORD_TYPE halfWidthA = boxA[2] / 2;
    const COORD_TYPE halfHeightA = boxA[3] / 2;
    const COORD_TYPE halfWidthB = boxB[2] / 2;
    const COORD_TYPE halfHeightB = boxB[3] / 2;

    const COORD_TYPE intersection_ymin = max(boxA[1] - halfHeightA, boxB[1] - halfHeightB);
    const COORD_TYPE intersection_xmin = max(boxA[0] - halfWidthA,  boxB[0] - halfWidthB);
    const COORD_TYPE intersection_ymax = min(boxA[1] + halfHeightA, boxB[1] + halfHeightB);
    const COORD_TYPE intersection_xmax = min(boxA[0] + halfWidthA,  boxB[0] + halfWidthB);
#endif

    if (areaA <= 0.0f || areaB <= 0.0f)
        return 0.0f;

    const float intersection_area = convert_float(max(intersection_xmax - intersection_xmin, TO_COORD_TYPE(0.f))) *
                                    convert_float(max(intersection_ymax - intersection_ymin, TO_COORD_TYPE(0.f)));
    const float union_area = areaA + areaB - intersection_area;
    return intersection_area / union_area;
}
#endif // ROTATION

inline float FUNC(scaleIOU)(float iou, float iou_threshold, float scale)
{
    if (iou <= iou_threshold || SOFT_NMS_SIGMA_VAL > 0.0f) {
        return exp(scale * iou * iou);
    } else {
        return 0.0f;
    }
}

inline void FUNC(swap_sbox_info)(__global SBOX_INFO* a, __global SBOX_INFO* b)
{
    SBOX_INFO temp = *a;
    *a = *b;
    *b = temp;
}

inline int FUNC(partition)(__global SBOX_INFO* arr, int l, int h)
{
    const INPUT1_TYPE pivotScore = arr[h].score;
    const ushort pivotBoxId = arr[h].boxId;
    int i = (l - 1);
    for (int j = l; j <= h - 1; j++) {
        if ((arr[j].score > pivotScore) || (arr[j].score == pivotScore && arr[j].boxId < pivotBoxId)) {
            i++;
            FUNC_CALL(swap_sbox_info)(&arr[i], &arr[j]);
        }
    }
    FUNC_CALL(swap_sbox_info)(&arr[i + 1], &arr[h]);
    return (i + 1);
}

inline void FUNC(bubbleSortIterative)(__global SBOX_INFO* arr, int l, int h)
{
    for (int i = 0; i < h-l; i++) {
        bool swapped = false;
        for (int j = l; j < h-i; j++) {
            if ((arr[j].score > arr[j+1].score) || (arr[j].score == arr[j+1].score && arr[j].boxId < arr[j+1].boxId)) {
                FUNC_CALL(swap_sbox_info)(&arr[j], &arr[j+1]);
                swapped = true;
            }
        }

        if (!swapped)
            break;
    }
}

inline void FUNC(quickSortIterative)(__global SBOX_INFO* arr, int l, int h)
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
        const int p = FUNC_CALL(partition)(arr, l, h);

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

inline int FUNC(initBoxList)(__global SBOX_INFO *outBoxes, int boxNum, const __global INPUT1_TYPE *scores, float score_threshold, short batchId, ushort classId)
{
    int count = 0;
    for (ushort i = 0; i < boxNum; ++i) {
        const INPUT1_TYPE score = scores[INPUT1_GET_INDEX(batchId, classId, i, 0)];
        if (convert_float(score) < score_threshold) continue;

        SBOX_INFO binfo;
        binfo.boxId = i;
        binfo.suppress_begin_index = 0;
        binfo.score = score;
        outBoxes[count] = binfo;
        ++count;
    }

    return count;
}

inline void FUNC(initOutputBoxList)(__global BOX_INFO *outBoxes, int boxNum, const __global INPUT1_TYPE *scores, __global OUTPUT_TYPE *output)
{
    for (int i = 0; i < boxNum; ++i) {
        outBoxes[i].batchId = output[OUTPUT_GET_INDEX(i, 0, 0, 0)];
        outBoxes[i].classId = output[OUTPUT_GET_INDEX(i, 1, 0, 0)];
        outBoxes[i].boxId = output[OUTPUT_GET_INDEX(i, 2, 0, 0)];
        outBoxes[i].score = scores[INPUT1_GET_INDEX(outBoxes[i].batchId, outBoxes[i].classId, outBoxes[i].boxId, 0)];
    }
}

inline void FUNC(swap)(__global BOX_INFO* a, __global BOX_INFO* b)
{
    BOX_INFO temp = *a;
    *a = *b;
    *b = temp;
}

inline void FUNC(sortOutputBoxList)(__global BOX_INFO *outSortedBoxes, int boxNum)
{
    for (int i = 0; i < boxNum - 1; ++i) {
        bool swapped = false;
        for (int j = 0; j < boxNum - i - 1; ++j) {
            if ((outSortedBoxes[j].score < outSortedBoxes[j+1].score) ||
                (outSortedBoxes[j].score == outSortedBoxes[j+1].score && outSortedBoxes[j].batchId > outSortedBoxes[j+1].batchId) ||
                (outSortedBoxes[j].score == outSortedBoxes[j+1].score && outSortedBoxes[j].batchId == outSortedBoxes[j+1].batchId &&
                 outSortedBoxes[j].classId > outSortedBoxes[j+1].classId) ||
                (outSortedBoxes[j].score == outSortedBoxes[j+1].score && outSortedBoxes[j].batchId == outSortedBoxes[j+1].batchId &&
                 outSortedBoxes[j].classId == outSortedBoxes[j+1].classId && outSortedBoxes[j].boxId > outSortedBoxes[j+1].boxId)) {
                FUNC_CALL(swap)(&outSortedBoxes[j], &outSortedBoxes[j+1]);
                swapped = true;
            }
        }

        // IF no two elements were swapped by inner loop, then break
        if (swapped == false)
            break;
    }
}


#ifdef NMS_STAGE_0
KERNEL (non_max_suppression_ref_stage_0)(
    const __global INPUT1_TYPE *scores
    , __global uchar *buffer0
    , __global int *buffer2
    #ifdef SCORE_THRESHOLD_TYPE
    , const __global SCORE_THRESHOLD_TYPE *score_threshold
    #endif
    )
{
    const int batchId = get_global_id(0);
    const int classId = get_global_id(1);
    const int box_gid = get_global_id(2);
    const int start_bid = box_gid * NUM_SCORE_PER_ITEM;
    const int end_bid = min(start_bid + NUM_SCORE_PER_ITEM, NUM_BOXES);

    __local char bit_mask[NUM_BIT_MASK];
    __local int block_num[NUM_SCORE_BLOCK];

    block_num[box_gid] = 0;
    {
        int mask_id = start_bid / 8;
        int total_block_selected_num = 0;
        for (int i = start_bid; i < end_bid; i += 8) {
            char mask = 0;
            for (int bi = 0; bi < 8; bi++) {
                if ((i + bi) >= NUM_BOXES)
                    break;

                if (convert_float(scores[INPUT1_GET_INDEX(batchId, classId, i, bi)]) <= SCORE_THRESHOLD_VAL)
                    continue;

                mask |= (1 << bi);
                total_block_selected_num++;
            }
            bit_mask[mask_id] = mask;
            mask_id++;
        }

        block_num[box_gid] = total_block_selected_num;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    {
        // first item of group
        if (box_gid == 0 && get_local_id(2) == 0) {
            int acc_num = 0;
            int total_sel_num = 0;
            for (int i = 0; i < NUM_SCORE_BLOCK; i++) {
                int n = block_num[i];
                block_num[i] = acc_num;
                acc_num += n;
            }
            buffer2[batchId * NUM_CLASSES + classId] = acc_num;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    {
        __global SBOX_INFO *sortedBoxList = (__global SBOX_INFO*)&buffer0[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];

        int write_offset = block_num[box_gid];

        int mask_id = start_bid / 8;
        for (int i = start_bid; i < end_bid; i += 8) {
            const char mask = bit_mask[mask_id];
            for (int bi = 0; bi < 8; bi++) {
                if ((mask & (1 << bi)) && (i + bi) < NUM_BOXES) {
                    SBOX_INFO binfo;
                    binfo.boxId = i + bi;
                    binfo.suppress_begin_index = 0;
                    binfo.score = scores[INPUT1_GET_INDEX(batchId, classId, i, bi)];
                    sortedBoxList[write_offset] = binfo;

                    write_offset++;
                }
            }
            mask_id++;
        }
    }
}
#endif /* NMS_STAGE_0 */

#ifdef NMS_STAGE_1

#if LOCAL_BATCH_NUM != 1
#error "The batch number of LWS should be 1."
#endif

KERNEL (non_max_suppression_ref_stage_1)(
    __global uchar *buffer0
    , __global int *buffer2
    )
{
    const int batchId = get_global_id(0);
    const int classId = get_global_id(1);
    const int workItemId = get_global_id(2);
    const int localClassId = get_local_id(1);
    __local int __range[LOCAL_CLASS_NUM][LOCAL_WORK_NUM * 2];
    const int kSortedBoxNum = buffer2[batchId * NUM_CLASSES + classId];
    __global SBOX_INFO *sortedBoxList = (__global SBOX_INFO*)&buffer0[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];
    if (workItemId == 0) {
        __range[localClassId][0] = 0;
        __range[localClassId][1] = kSortedBoxNum - 1;
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
                const int pivot = FUNC_CALL(partition)(sortedBoxList, begin_id, end_id);
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
        FUNC_CALL(quickSortIterative)(sortedBoxList, begin_id, end_id);
    }
}
#endif /* NMS_STAGE_1 */

#ifdef NMS_STAGE_2
KERNEL (non_max_suppression_ref_stage_2)(
    const __global INPUT0_TYPE *boxes
    , __global uchar *buffer0
    , __global uchar *buffer1
    , __global int *buffer2
    #ifdef NUM_SELECT_PER_CLASS_TYPE
    , const __global NUM_SELECT_PER_CLASS_TYPE *num_select_per_class
    #endif
    #ifdef IOU_THRESHOLD_TYPE
    , const __global IOU_THRESHOLD_TYPE *iou_threshold
    #endif
    #ifdef SCORE_THRESHOLD_TYPE
    , const __global SCORE_THRESHOLD_TYPE *score_threshold
    #endif
    #ifdef SOFT_NMS_SIGMA_TYPE
    , const __global SOFT_NMS_SIGMA_TYPE *soft_nms_sigma
    #endif
    )
{
    const short batchId = get_global_id(0);
    const ushort classId = get_global_id(1);

    float scale = 0.0f;
    #ifndef ROTATION
    if (SOFT_NMS_SIGMA_VAL > 0.0f) {
        scale = -0.5f / SOFT_NMS_SIGMA_VAL;
    }
    #endif

    __global SBOX_INFO *sortedBoxList = (__global SBOX_INFO*)&buffer0[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];
    const int kSortedBoxNum = buffer2[batchId * NUM_CLASSES + classId];

    __global BOX_INFO *selectedBoxList = (__global BOX_INFO*)&buffer1[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];
    int selectedBoxNum = 0;
    const int kNumSelectPerClass = NUM_SELECT_PER_CLASS_VAL;
    int i = 0;
    while (i < kSortedBoxNum && selectedBoxNum < kNumSelectPerClass) {
        SBOX_INFO next_candidate = sortedBoxList[i];
        INPUT1_TYPE original_score = next_candidate.score;
        const COORD_TYPE_4 next_candidate_coord = FUNC_CALL(getBoxCoords)(boxes, batchId, next_candidate.boxId);
        #ifdef ROTATION
        const INPUT0_TYPE next_candidate_angle = boxes[INPUT0_GET_INDEX(batchId, next_candidate.boxId, 4, 0)];
        #endif

        ++i;

        bool should_hard_suppress = false;
        for (int j = selectedBoxNum - 1; j >= next_candidate.suppress_begin_index; --j) {
            const COORD_TYPE_4 selected_box_coord = FUNC_CALL(getBoxCoords)(boxes, batchId, selectedBoxList[j].boxId);
            #ifdef ROTATION
            const INPUT0_TYPE selected_box_angle = boxes[INPUT0_GET_INDEX(batchId, selectedBoxList[j].boxId, 4, 0)];
            const float iou = FUNC_CALL(intersectionOverUnion)(next_candidate_coord, next_candidate_angle,
                    selected_box_coord, selected_box_angle);
            #else
            const float iou = FUNC_CALL(intersectionOverUnion)(next_candidate_coord, selected_box_coord);
            #endif
            next_candidate.score *= FUNC_CALL(scaleIOU)(iou, IOU_THRESHOLD_VAL, scale);

            if (iou >= IOU_THRESHOLD_VAL && !(SOFT_NMS_SIGMA_VAL > 0.0f)) {
                should_hard_suppress = true;
                break;
            }

            if (convert_float(next_candidate.score) <= SCORE_THRESHOLD_VAL) {
                break;
            }
        }

        next_candidate.suppress_begin_index = selectedBoxNum;

        if (!should_hard_suppress) {
            if (next_candidate.score == original_score) {
                BOX_INFO binfo;
                binfo.batchId = batchId;
                binfo.classId = classId;
                binfo.boxId = next_candidate.boxId;
                binfo.score = next_candidate.score;
                selectedBoxList[selectedBoxNum] = binfo;
                ++selectedBoxNum;

                continue;
            }

            if (convert_float(next_candidate.score) > SCORE_THRESHOLD_VAL) {
                --i;
                sortedBoxList[i] = next_candidate;
                FUNC_CALL(quickSortIterative)(sortedBoxList, i, kSortedBoxNum - 1);
            }
        }
    }

    // Set pad value to indicate the end of selected box list.
    if (selectedBoxNum < NUM_BOXES) {
        int b = selectedBoxNum;
        #ifdef REUSE_INTERNAL_BUFFER
            for (; b < NUM_BOXES; ++b) {
                selectedBoxList[b].batchId = -1;
            }
        #else
            selectedBoxList[b].batchId = -1;
        #endif
    }
}
#endif /* NMS_STAGE_2 */

#ifdef NMS_STAGE_3
KERNEL (non_max_suppression_ref_stage_3)(
    __global OUTPUT_TYPE *output
    , __global uchar *buffer1
    , __global uchar *buffer2
#ifdef OUTPUT1_TYPE
    , __global OUTPUT1_TYPE *selected_scores
#endif
#ifdef OUTPUT2_TYPE
    , __global OUTPUT2_TYPE *valid_outputs
#endif
    )
{
    int outputIdx = 0;
    __global BOX_INFO *sortedBoxList = (__global BOX_INFO*)&buffer2[0];
    for (short batchId = 0; batchId < NUM_BATCHES; batchId++) {
        for (ushort classId = 0; classId < NUM_CLASSES; classId++) {
            __global BOX_INFO *selectedBoxList = (__global BOX_INFO*)&buffer1[(batchId * NUM_CLASSES + classId) * BUFFER_STRIDE];
            for (int i = 0; i < NUM_BOXES; i++) {
                if (selectedBoxList[i].batchId > -1) {
                    sortedBoxList[outputIdx] = selectedBoxList[i];
                    outputIdx++;
                } else {
                    break;
                }
            }
        }
    }

#if SORT_RESULT_DESCENDING == 1
    FUNC_CALL(sortOutputBoxList)(sortedBoxList, outputIdx);
#endif

    unroll_for (int i = 0; i < outputIdx; i++) {
        output[OUTPUT_GET_INDEX(i, 0, 0, 0)] = sortedBoxList[i].batchId;
        output[OUTPUT_GET_INDEX(i, 1, 0, 0)] = sortedBoxList[i].classId;
        output[OUTPUT_GET_INDEX(i, 2, 0, 0)] = sortedBoxList[i].boxId;
    }

    // Padding
    unroll_for (int i = outputIdx; i < OUTPUT_NUM; i++) {
        output[OUTPUT_GET_INDEX(i, 0, 0, 0)] = -1;
        output[OUTPUT_GET_INDEX(i, 1, 0, 0)] = -1;
        output[OUTPUT_GET_INDEX(i, 2, 0, 0)] = -1;
    }

#ifdef OUTPUT1_TYPE
    unroll_for (int i = 0; i < outputIdx; i++) {
        selected_scores[OUTPUT1_GET_INDEX(i, 0, 0, 0)] = TO_OUTPUT1_TYPE(sortedBoxList[i].batchId);
        selected_scores[OUTPUT1_GET_INDEX(i, 1, 0, 0)] = TO_OUTPUT1_TYPE(sortedBoxList[i].classId);
        selected_scores[OUTPUT1_GET_INDEX(i, 2, 0, 0)] = TO_OUTPUT1_TYPE(sortedBoxList[i].score);
    }

    // Padding
    unroll_for (int i = outputIdx; i < OUTPUT_NUM; i++) {
        selected_scores[OUTPUT1_GET_INDEX(i, 0, 0, 0)] = -1;
        selected_scores[OUTPUT1_GET_INDEX(i, 1, 0, 0)] = -1;
        selected_scores[OUTPUT1_GET_INDEX(i, 2, 0, 0)] = -1;
    }
#endif

#ifdef OUTPUT2_TYPE
    valid_outputs[OUTPUT2_GET_INDEX(0, 0, 0, 0)] = TO_OUTPUT2_TYPE(outputIdx);
#endif  // OUTPUT1_TYPE
}
#endif  /* NMS_STAGE_3 */
