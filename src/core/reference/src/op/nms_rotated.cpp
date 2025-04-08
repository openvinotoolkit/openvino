// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/nms_rotated.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>

#include "openvino/reference/nms_rotated_util.hpp"

namespace ov {
namespace reference {
namespace nms_detail {
using iou_rotated::RotatedBox;
static float rotated_intersection_over_union(const RotatedBox& boxI, const RotatedBox& boxJ) {
    const auto intersection = iou_rotated::rotated_boxes_intersection(boxI, boxJ);
    const auto areaI = boxI.w * boxI.h;
    const auto areaJ = boxJ.w * boxJ.h;

    if (areaI <= 0.0f || areaJ <= 0.0f) {
        return 0.0f;
    }

    const auto union_area = areaI + areaJ - intersection;
    return intersection / union_area;
}

struct SelectedIndex {
    SelectedIndex(int64_t batch_idx, int64_t class_idx, int64_t box_idx)
        : batch_index(batch_idx),
          class_index(class_idx),
          box_index(box_idx) {}

    SelectedIndex() = default;

    int64_t batch_index = 0;
    int64_t class_index = 0;
    int64_t box_index = 0;
};

struct SelectedScore {
    SelectedScore(float batch_idx, float class_idx, float score)
        : batch_index{batch_idx},
          class_index{class_idx},
          box_score{score} {}

    SelectedScore() = default;

    float batch_index = 0.0f;
    float class_index = 0.0f;
    float box_score = 0.0f;
};

struct BoxInfo {
    BoxInfo(const RotatedBox& r, int64_t idx, float sc, int64_t suppress_idx, int64_t batch_idx, int64_t class_idx)
        : box{r},
          index{idx},
          suppress_begin_index{suppress_idx},
          batch_index{batch_idx},
          class_index{class_idx},
          score{sc} {}

    BoxInfo() = default;

    inline bool operator<(const BoxInfo& rhs) const {
        return score < rhs.score || (score == rhs.score && index > rhs.index);
    }

    RotatedBox box;
    int64_t index = 0;
    int64_t suppress_begin_index = 0;
    int64_t batch_index = 0;
    int64_t class_index = 0;
    float score = 0.0f;
};
}  // namespace nms_detail

void nms_rotated(const float* boxes_data,
                 const Shape& boxes_data_shape,
                 const float* scores_data,
                 const Shape& scores_data_shape,
                 int64_t max_output_boxes_per_class,
                 float iou_threshold,
                 float score_threshold,
                 float soft_nms_sigma,
                 int64_t* selected_indices,
                 const Shape& selected_indices_shape,
                 float* selected_scores,
                 const Shape& selected_scores_shape,
                 int64_t* valid_outputs,
                 const bool sort_result_descending,
                 const bool clockwise) {
    using iou_rotated::RotatedBox;
    using nms_detail::BoxInfo;
    using nms_detail::SelectedIndex;
    using nms_detail::SelectedScore;

    // The code for softsigma is kept to simplify unification with NMS code,
    // but for NMSRotated softsigma is not supported (always 0.0);
    float scale = 0.0f;
    bool soft_nms = false;
    if (soft_nms_sigma > 0.0f) {
        scale = -0.5f / soft_nms_sigma;
        soft_nms = true;
    }

    auto get_score_scale = [iou_threshold, scale, soft_nms](float iou) {
        const float weight = std::exp(scale * iou * iou);
        return (soft_nms || iou <= iou_threshold) ? weight : 0.0f;
    };

    // boxes shape: {num_batches, num_boxes, 5}
    // scores shape: {num_batches, num_classes, num_boxes}
    int64_t num_batches = static_cast<int64_t>(scores_data_shape[0]);
    int64_t num_classes = static_cast<int64_t>(scores_data_shape[1]);
    int64_t num_boxes = static_cast<int64_t>(boxes_data_shape[1]);

    SelectedIndex* selected_indices_ptr = reinterpret_cast<SelectedIndex*>(selected_indices);
    SelectedScore* selected_scores_ptr = reinterpret_cast<SelectedScore*>(selected_scores);

    size_t boxes_per_class = static_cast<size_t>(max_output_boxes_per_class);

    std::vector<BoxInfo> filteredBoxes;

    for (int64_t batch = 0; batch < num_batches; batch++) {
        const float* boxesPtr = boxes_data + batch * num_boxes * 5;
        const RotatedBox* r = reinterpret_cast<const RotatedBox*>(boxesPtr);

        for (int64_t class_idx = 0; class_idx < num_classes; class_idx++) {
            const float* scoresPtr = scores_data + batch * (num_classes * num_boxes) + class_idx * num_boxes;

            std::vector<BoxInfo> candidate_boxes;
            candidate_boxes.reserve(num_boxes);

            for (int64_t box_idx = 0; box_idx < num_boxes; box_idx++) {
                if (scoresPtr[box_idx] > score_threshold) {
                    candidate_boxes.emplace_back(r[box_idx], box_idx, scoresPtr[box_idx], 0, batch, class_idx);
                    // Convert counterclockwise to clockwise
                    if (!clockwise) {
                        candidate_boxes.back().box.a *= -1.f;
                    }
                }
            }

            std::priority_queue<BoxInfo> sorted_boxes(std::less<BoxInfo>(), std::move(candidate_boxes));

            std::vector<BoxInfo> selected;
            // Get the next box with top score, filter by iou_threshold

            BoxInfo next_candidate;
            float original_score;

            while (!sorted_boxes.empty() && selected.size() < boxes_per_class) {
                next_candidate = sorted_boxes.top();
                original_score = next_candidate.score;
                sorted_boxes.pop();

                bool should_hard_suppress = false;
                for (int64_t j = static_cast<int64_t>(selected.size()) - 1; j >= next_candidate.suppress_begin_index;
                     --j) {
                    // The main difference between NMS and NMSRotated is the calculation of iou for rotated boxes
                    float iou = nms_detail::rotated_intersection_over_union(next_candidate.box, selected[j].box);
                    next_candidate.score *= get_score_scale(iou);

                    if ((iou > iou_threshold) && !soft_nms) {
                        should_hard_suppress = true;
                        break;
                    }

                    if (next_candidate.score <= score_threshold) {
                        break;
                    }
                }

                next_candidate.suppress_begin_index = selected.size();

                if (!should_hard_suppress) {
                    if (next_candidate.score == original_score) {
                        selected.push_back(next_candidate);
                        continue;
                    }
                    if (next_candidate.score > score_threshold) {
                        sorted_boxes.push(next_candidate);
                    }
                }
            }

            for (const auto& box_info : selected) {
                filteredBoxes.push_back(box_info);
            }
        }
    }

    if (sort_result_descending) {
        std::stable_sort(filteredBoxes.begin(), filteredBoxes.end(), [](const BoxInfo& lhs, const BoxInfo& rhs) {
            return (lhs.score > rhs.score) ||
                   ((lhs.score == rhs.score) && (std::tie(lhs.batch_index, lhs.class_index, lhs.index) <
                                                 std::tie(rhs.batch_index, rhs.class_index, rhs.index)));
        });
    }

    size_t max_num_of_selected_indices = selected_indices_shape[0];
    size_t output_size = std::min(filteredBoxes.size(), max_num_of_selected_indices);

    *valid_outputs = output_size;

    size_t idx;
    for (idx = 0; idx < output_size; idx++) {
        const auto& box_info = filteredBoxes[idx];
        selected_indices_ptr[idx] = SelectedIndex{box_info.batch_index, box_info.class_index, box_info.index};
        selected_scores_ptr[idx] = SelectedScore{static_cast<float>(box_info.batch_index),
                                                 static_cast<float>(box_info.class_index),
                                                 box_info.score};
    }

    for (; idx < max_num_of_selected_indices; idx++) {
        selected_indices_ptr[idx] = SelectedIndex{0, 0, 0};
        selected_scores_ptr[idx] = SelectedScore{0.0f, 0.0f, 0.0f};
    }
}

}  // namespace reference
}  // namespace ov
