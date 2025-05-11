// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (c) Facebook, Inc. and its affiliates.
// The implementation for rotated boxes intersection is based on the code from:
// https://github.com/facebookresearch/detectron2/blob/v0.6/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_utils.h

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>

namespace ov {
namespace reference {
namespace iou_rotated {
struct RotatedBox {
    float x_ctr, y_ctr, w, h, a;
};
struct Point2D {
    float x, y;
    Point2D(const float px = 0.f, const float py = 0.f) : x(px), y(py) {}
    Point2D operator+(const Point2D& p) const {
        return Point2D(x + p.x, y + p.y);
    }
    Point2D& operator+=(const Point2D& p) {
        x += p.x;
        y += p.y;
        return *this;
    }
    Point2D operator-(const Point2D& p) const {
        return Point2D(x - p.x, y - p.y);
    }
    Point2D operator*(const float coeff) const {
        return Point2D(x * coeff, y * coeff);
    }
};

static inline float dot_2d(const Point2D& A, const Point2D& B) {
    return A.x * B.x + A.y * B.y;
}

static inline float cross_2d(const Point2D& A, const Point2D& B) {
    return A.x * B.y - B.x * A.y;
}

// Calculate box vertices rotated by angle (clockwise) over the box center
static inline void get_rotated_vertices(const RotatedBox& box, Point2D (&pts)[4]) {
    // M_PI / 180. == 0.01745329251
    auto theta = box.a;  // angle already in radians
    auto cosTheta2 = std::cos(theta) * 0.5f;
    auto sinTheta2 = std::sin(theta) * 0.5f;

    // y: top --> down; x: left --> right
    // Left-Down
    pts[0].x = box.x_ctr - sinTheta2 * box.h - cosTheta2 * box.w;
    pts[0].y = box.y_ctr + cosTheta2 * box.h - sinTheta2 * box.w;
    // Left-Top
    pts[1].x = box.x_ctr + sinTheta2 * box.h - cosTheta2 * box.w;
    pts[1].y = box.y_ctr - cosTheta2 * box.h - sinTheta2 * box.w;
    // Right-Top
    pts[2].x = 2 * box.x_ctr - pts[0].x;
    pts[2].y = 2 * box.y_ctr - pts[0].y;
    // Right-Down
    pts[3].x = 2 * box.x_ctr - pts[1].x;
    pts[3].y = 2 * box.y_ctr - pts[1].y;
}

// Find points defining area of the boxes intersection:
// - Find all intersection points between edges of the boxes
// - Find all corners of box1 within area of box2, and all corners of box2 within area of box1
static inline int get_intersection_points(const Point2D (&pts1)[4],
                                          const Point2D (&pts2)[4],
                                          Point2D (&intersections)[24]) {
    // Line vector
    // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
    Point2D vec1[4], vec2[4];
    for (int i = 0; i < 4; i++) {
        vec1[i] = pts1[(i + 1) % 4] - pts1[i];
        vec2[i] = pts2[(i + 1) % 4] - pts2[i];
    }

    // Line test - test all line combos for intersection
    int num = 0;  // number of intersections
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            // Solve for 2x2 Ax=b
            float det = cross_2d(vec2[j], vec1[i]);

            // This takes care of parallel lines
            if (std::abs(det) <= 1e-14f) {
                continue;
            }

            auto vec12 = pts2[j] - pts1[i];

            auto t1 = cross_2d(vec2[j], vec12) / det;
            auto t2 = cross_2d(vec1[i], vec12) / det;

            if (t1 >= 0.0f && t1 <= 1.0f && t2 >= 0.0f && t2 <= 1.0f) {
                intersections[num++] = pts1[i] + vec1[i] * t1;
            }
        }
    }

    // Check for vertices of rect1 inside rect2
    {
        const auto& AB = vec2[0];
        const auto& DA = vec2[3];
        auto ABdotAB = dot_2d(AB, AB);
        auto ADdotAD = dot_2d(DA, DA);
        for (int i = 0; i < 4; i++) {
            // assume ABCD is the rectangle, and P is the point to be judged
            // P is inside ABCD iff. P's projection on AB lies within AB
            // and P's projection on AD lies within AD

            auto AP = pts1[i] - pts2[0];

            auto APdotAB = dot_2d(AP, AB);
            auto APdotAD = -dot_2d(AP, DA);

            if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) && (APdotAD <= ADdotAD)) {
                intersections[num++] = pts1[i];
            }
        }
    }

    // Reverse the check - check for vertices of rect2 inside rect1
    {
        const auto& AB = vec1[0];
        const auto& DA = vec1[3];
        auto ABdotAB = dot_2d(AB, AB);
        auto ADdotAD = dot_2d(DA, DA);
        for (int i = 0; i < 4; i++) {
            auto AP = pts2[i] - pts1[0];

            auto APdotAB = dot_2d(AP, AB);
            auto APdotAD = -dot_2d(AP, DA);

            if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) && (APdotAD <= ADdotAD)) {
                intersections[num++] = pts2[i];
            }
        }
    }

    return num;
}

static inline int convex_hull_graham(const Point2D (&p)[24],
                                     const int num_in,
                                     Point2D (&q)[24],
                                     bool shift_to_zero = false) {
    assert(num_in >= 2);

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
    auto& start = p[t];  // starting point

    // Step 2:
    // Subtract starting point from every points (for sorting in the next step)
    for (int i = 0; i < num_in; i++) {
        q[i] = p[i] - start;
    }

    // Swap the starting point to position 0
    std::swap(q[t], q[0]);

    // Step 3:
    // Sort point 1 ~ num_in according to their relative cross-product values
    // (essentially sorting according to angles)
    // If the angles are the same, sort according to their distance to origin
    float dist[24];
    for (int i = 0; i < num_in; i++) {
        dist[i] = dot_2d(q[i], q[i]);
    }

    std::sort(q + 1, q + num_in, [](const Point2D& A, const Point2D& B) -> bool {
        float temp = cross_2d(A, B);
        if (std::abs(temp) < 1e-6f) {
            return dot_2d(A, A) < dot_2d(B, B);
        } else {
            return temp > 0;
        }
    });
    // compute distance to origin after sort, since the points are now different.
    for (int i = 0; i < num_in; i++) {
        dist[i] = dot_2d(q[i], q[i]);
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
        q[0] = p[t];
        return 1;
    }
    q[1] = q[k];
    int m = 2;  // 2 points in the stack
    // Step 5:
    // Finally we can start the scanning process.
    // When a non-convex relationship between the 3 points is found
    // (either concave shape or duplicated points),
    // we pop the previous point from the stack
    // until the 3-point relationship is convex again, or
    // until the stack only contains two points
    for (int i = k + 1; i < num_in; i++) {
        while (m > 1 && cross_2d(q[i] - q[m - 2], q[m - 1] - q[m - 2]) >= 0) {
            m--;
        }
        q[m++] = q[i];
    }

    // Step 6 (Optional):
    // In general sense we need the original coordinates, so we
    // need to shift the points back (reverting Step 2)
    // But if we're only interested in getting the area/perimeter of the shape
    // We can simply return.
    if (!shift_to_zero) {
        for (int i = 0; i < m; i++) {
            q[i] += start;
        }
    }

    return m;
}

static inline float polygon_area(const Point2D (&q)[24], const int& m) {
    if (m <= 2) {
        return 0.f;
    }

    float area = 0.f;
    for (int i = 1; i < m - 1; i++) {
        area += std::abs(cross_2d(q[i] - q[0], q[i + 1] - q[0]));
    }

    return area / 2.0f;
}

static inline float rotated_boxes_intersection(const RotatedBox& box1, const RotatedBox& box2) {
    // There are up to 4 x 4 + 4 + 4 = 24 intersections (including dups) returned
    // from get_intersection_points
    Point2D intersectPts[24], orderedPts[24];

    Point2D pts1[4];
    Point2D pts2[4];
    get_rotated_vertices(box1, pts1);
    get_rotated_vertices(box2, pts2);

    // Find points defining area of the boxes intersection
    int num = get_intersection_points(pts1, pts2, intersectPts);

    if (num <= 2) {
        return 0.f;
    }

    // Convex Hull to order the intersection points in clockwise order and find
    // the contour area.
    int num_convex = convex_hull_graham(intersectPts, num, orderedPts, true);
    return polygon_area(orderedPts, num_convex);
}

}  // namespace iou_rotated

}  // namespace reference
}  // namespace ov
