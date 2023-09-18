// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/nms_rotated.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>

#include "ngraph/op/non_max_suppression.hpp"
#include "ngraph/shape.hpp"
#include "openvino/reference/non_max_suppression.hpp"

namespace ov {
namespace reference {
namespace nms_rotated {

namespace {

static float rotatedintersectionOverUnion(const RotatedBox& boxI, const RotatedBox& boxJ) {
    const auto intersection = rotated_boxes_intersection(boxI, boxJ);
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
}  // namespace

void non_max_suppression(const float* boxes_data,
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
                         const bool sort_result_descending) {
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
        RotatedBox* r = reinterpret_cast<RotatedBox*>(const_cast<float*>(boxesPtr));

        for (int64_t class_idx = 0; class_idx < num_classes; class_idx++) {
            const float* scoresPtr = scores_data + batch * (num_classes * num_boxes) + class_idx * num_boxes;

            std::vector<BoxInfo> candidate_boxes;
            candidate_boxes.reserve(num_boxes);

            for (int64_t box_idx = 0; box_idx < num_boxes; box_idx++) {
                if (scoresPtr[box_idx] > score_threshold) {
                    candidate_boxes.emplace_back(r[box_idx], box_idx, scoresPtr[box_idx], 0, batch, class_idx);
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
                    float iou = rotatedintersectionOverUnion(next_candidate.box, selected[j].box);
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
        std::reverse(filteredBoxes.begin(), filteredBoxes.end());
    }

    size_t max_num_of_selected_indices = selected_indices_shape[0];
    size_t output_size = std::min(filteredBoxes.size(), max_num_of_selected_indices);

    *valid_outputs = output_size;

    size_t idx;
    for (idx = 0; idx < output_size; idx++) {
        const auto& box_info = filteredBoxes[idx];
        SelectedIndex selected_index{box_info.batch_index, box_info.class_index, box_info.index};
        SelectedScore selected_score{static_cast<float>(box_info.batch_index),
                                     static_cast<float>(box_info.class_index),
                                     box_info.score};

        selected_indices_ptr[idx] = selected_index;
        selected_scores_ptr[idx] = selected_score;
    }

    SelectedIndex selected_index_filler{0, 0, 0};
    SelectedScore selected_score_filler{0.0f, 0.0f, 0.0f};
    for (; idx < max_num_of_selected_indices; idx++) {
        selected_indices_ptr[idx] = selected_index_filler;
        selected_scores_ptr[idx] = selected_score_filler;
    }
}

OPENVINO_SUPPRESS_DEPRECATED_START
void nms_postprocessing(const HostTensorVector& outputs,
                        const ngraph::element::Type output_type,
                        const std::vector<int64_t>& selected_indices,
                        const std::vector<float>& selected_scores,
                        int64_t valid_outputs,
                        const ngraph::element::Type selected_scores_type) {
    outputs[0]->set_element_type(output_type);
    outputs[0]->set_shape(Shape{static_cast<size_t>(valid_outputs), 3});

    size_t num_of_outputs = outputs.size();

    if (num_of_outputs >= 2) {
        outputs[1]->set_element_type(selected_scores_type);
        outputs[1]->set_shape(Shape{static_cast<size_t>(valid_outputs), 3});
    }

    if (num_of_outputs >= 3) {
        outputs[2]->set_element_type(output_type);
        outputs[2]->set_shape(Shape{1});
    }

    size_t selected_size = valid_outputs * 3;

    if (output_type == ngraph::element::i64) {
        int64_t* indices_ptr = outputs[0]->get_data_ptr<int64_t>();
        memcpy(indices_ptr, selected_indices.data(), selected_size * sizeof(int64_t));
    } else {
        int32_t* indices_ptr = outputs[0]->get_data_ptr<int32_t>();
        for (size_t i = 0; i < selected_size; ++i) {
            indices_ptr[i] = static_cast<int32_t>(selected_indices[i]);
        }
    }

    if (num_of_outputs < 2) {
        return;
    }

    size_t selected_scores_size = selected_scores.size();

    switch (selected_scores_type) {
    case element::Type_t::bf16: {
        bfloat16* scores_ptr = outputs[1]->get_data_ptr<bfloat16>();
        for (size_t i = 0; i < selected_scores_size; ++i) {
            scores_ptr[i] = bfloat16(selected_scores[i]);
        }
    } break;
    case element::Type_t::f16: {
        float16* scores_ptr = outputs[1]->get_data_ptr<float16>();
        for (size_t i = 0; i < selected_scores_size; ++i) {
            scores_ptr[i] = float16(selected_scores[i]);
        }
    } break;
    case element::Type_t::f32: {
        float* scores_ptr = outputs[1]->get_data_ptr<float>();
        memcpy(scores_ptr, selected_scores.data(), selected_size * sizeof(float));
    } break;
    default:;
    }

    if (num_of_outputs < 3) {
        return;
    }

    if (output_type == ngraph::element::i64) {
        int64_t* valid_outputs_ptr = outputs[2]->get_data_ptr<int64_t>();
        *valid_outputs_ptr = valid_outputs;
    } else {
        int32_t* valid_outputs_ptr = outputs[2]->get_data_ptr<int32_t>();
        *valid_outputs_ptr = static_cast<int32_t>(valid_outputs);
    }
}
}  // namespace nms_rotated

}  // namespace reference
}  // namespace ov
