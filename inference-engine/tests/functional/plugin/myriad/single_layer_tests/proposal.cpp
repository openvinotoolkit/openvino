// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/proposal.hpp"
#include "common_test_utils/test_constants.hpp"
#include "precision_utils.h"
#include "common_test_utils/test_constants.hpp"

// additional features for developer testing
//#define PROPOSAL_TESTS_LOGGING
const bool compareOutputScoresSoft = true; // do compare scores against reference with threshold
const float scoresThreshold = 0.01f; // threshold for soft scores comparison
const bool compareOutputScores = false; // do compare scores output against reference without threshold

using namespace InferenceEngine;
using namespace PrecisionUtils;

/**
* "IoU = intersection area / union area" of two boxes A, B
*   A, B: 4-dim array (x1, y1, x2, y2)
*/

static float check_iou(const float* A, const float* B) {
    if (A[0] > B[2] || A[1] > B[3] || A[2] < B[0] || A[3] < B[1]) {
        return 0.0f;
    } else {
        // overlapped region (= box)
        const float x1 = std::max(A[0], B[0]);
        const float y1 = std::max(A[1], B[1]);
        const float x2 = std::min(A[2], B[2]);
        const float y2 = std::min(A[3], B[3]);

        // intersection area
        const float width = std::max(0.0f, x2 - x1 + 1.0f);
        const float height = std::max(0.0f, y2 - y1 + 1.0f);
        const float area = width * height;

        // area of A, B
        const float A_area = (A[2] - A[0] + 1.0f) * (A[3] - A[1] + 1.0f);
        const float B_area = (B[2] - B[0] + 1.0f) * (B[3] - B[1] + 1.0f);

        // IoU
        return area / (A_area + B_area - area);
    }
}

static float check_iou_normalized(const float* A, const float* B) {
    if (A[0] > B[2] || A[1] > B[3] || A[2] < B[0] || A[3] < B[1]) {
        return 0.0f;
    } else {
        // overlapped region (= box)
        const float x1 = std::max(A[0], B[0]);
        const float y1 = std::max(A[1], B[1]);
        const float x2 = std::min(A[2], B[2]);
        const float y2 = std::min(A[3], B[3]);

        // intersection area
        const float width = std::max(0.0f, x2 - x1);
        const float height = std::max(0.0f, y2 - y1);
        const float area = width * height;

        // area of A, B
        const float A_area = (A[2] - A[0]) * (A[3] - A[1]);
        const float B_area = (B[2] - B[0]) * (B[3] - B[1]);

        // IoU
        return area / (A_area + B_area - area);
    }
}

static std::size_t get_num_rois(const ie_fp16* array, std::size_t size) {
    std::size_t count = 0;
    while (count < size && f16tof32(array[count]) != -1.f)
        count += 5;
    return count / 5;
}

typedef std::vector<std::vector<int>> Graph;

class GraphComparison {
public:
    int calculateMatchingCount(const ie_fp16* reference, std::size_t count_reference,
                               const ie_fp16* optimized, std::size_t count_optimized,
                               const ie_fp16* reference_scores, const ie_fp16* output_scores,
                               bool normalized, bool withOutputScores) {
        validateData(reference, count_reference, optimized, count_optimized);

        //edges from vertexes of optimized boxes to vertexes of reference boxes
        Graph graph {make_graph(reference, count_reference, optimized, count_optimized,
                                reference_scores, output_scores,
                                normalized, withOutputScores)};

        int matching_count = 0;
        std::vector<bool> used(count_optimized, false);
        std::vector<int> mt(count_reference, -1); // matching, -1 means there is no matching
        for (std::size_t vertex = 0; vertex < count_optimized; ++vertex) {
            if (try_kuhn(used, mt, graph, vertex)) {
                used.assign(count_optimized, false);
                ++matching_count;
            }
        }
        return matching_count;
    }

    void set_threshold(float threshold) {
        _threshold = threshold;
    }

private:
    float _threshold;

    Graph make_graph(const ie_fp16* reference, std::size_t count_reference,
                     const ie_fp16* optimized, std::size_t count_optimized,
                     const ie_fp16* reference_scores, const ie_fp16* output_scores,
                     bool normalized, bool withOutputScores) {
        Graph g(count_optimized);
        for (std::size_t i = 0; i < count_optimized; ++i) {
            float out_values[4]{};
            for (int k = 0; k < 4; k++)
                    out_values[k] = f16tof32(optimized[i * 5 + k + 1]);
            for (std::size_t j = 0; j < count_reference; ++j) {
                float gt_values[4]{};
                for (int k = 0; k < 4; k++)
                    gt_values[k] = f16tof32(reference[j * 5 + k + 1]);

                bool isScoresEqual = true;
                if (compareOutputScores && withOutputScores)
                    isScoresEqual = (output_scores[i] == reference_scores[j]);

                if (isScoresEqual) {
                    if (!normalized && check_iou(out_values, gt_values) >= _threshold)
                        g[i].push_back(j);
                    else if (normalized && check_iou_normalized(out_values, gt_values) >= _threshold)
                        g[i].push_back(j);
                }
            }
        }
        return g;
    }

    bool try_kuhn(std::vector<bool>& used,
                  std::vector<int>& mt,
                  Graph& graph,
                  int cur_vertex) {
        if (used[cur_vertex]) return false;
        used[cur_vertex] = true;
        for (auto to : graph[cur_vertex]) {
            if (mt[to] == -1) {
                mt[to] = cur_vertex;
                return true;
            }
        }
        for (auto to : graph[cur_vertex]) {
            if (try_kuhn(used, mt, graph, mt[to])) {
                mt[to] = cur_vertex;
                return true;
            }
        }
        return false;
    }

    // if x1 > x2 or y1 > y2, then we can assume that the data is incorrect
    void validateData(const ie_fp16* reference, std::size_t count_reference,
                      const ie_fp16* optimized, std::size_t count_optimized) {
        for (std::size_t i = 0; i < count_reference; i++) {
            assert(reference[i * 5 + 1] <= reference[i * 5 + 3] &&
                   "Incorrect data: x1 > x2 in reference boxes");
            assert(reference[i * 5 + 2] <= reference[i * 5 + 4] &&
                   "Incorrect data: y1 > y2 in reference boxes");
        }

        for (std::size_t i = 0; i < count_optimized; i++) {
            assert(optimized[i * 5 + 1] <= optimized[i * 5 + 3] &&
                   "Incorrect data: x1 > x2 in optimized boxes");
            assert(optimized[i * 5 + 2] <= optimized[i * 5 + 4] &&
                   "Incorrect data: y1 > y2 in optimized boxes");
        }
    }
};

