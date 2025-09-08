// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/non_max_suppression.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>

#include "openvino/reference/non_max_suppression.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace reference {
namespace {
struct Rectangle {
    Rectangle(float y_left, float x_left, float y_right, float x_right)
        : y1{y_left},
          x1{x_left},
          y2{y_right},
          x2{x_right} {}

    Rectangle() = default;

    float y1 = 0.0f;
    float x1 = 0.0f;
    float y2 = 0.f;
    float x2 = 0.0f;
};

static float intersectionOverUnion(const Rectangle& boxI, const Rectangle& boxJ) {
    float areaI = (boxI.y2 - boxI.y1) * (boxI.x2 - boxI.x1);
    float areaJ = (boxJ.y2 - boxJ.y1) * (boxJ.x2 - boxJ.x1);

    if (areaI <= 0.0f || areaJ <= 0.0f) {
        return 0.0f;
    }

    float intersection_ymin = std::max(boxI.y1, boxJ.y1);
    float intersection_xmin = std::max(boxI.x1, boxJ.x1);
    float intersection_ymax = std::min(boxI.y2, boxJ.y2);
    float intersection_xmax = std::min(boxI.x2, boxJ.x2);

    float intersection_area =
        std::max(intersection_ymax - intersection_ymin, 0.0f) * std::max(intersection_xmax - intersection_xmin, 0.0f);

    return intersection_area / (areaI + areaJ - intersection_area);
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
    BoxInfo(const Rectangle& r, int64_t idx, float sc, int64_t suppress_idx, int64_t batch_idx, int64_t class_idx)
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

    Rectangle box;
    int64_t index = 0;
    int64_t suppress_begin_index = 0;
    int64_t batch_index = 0;
    int64_t class_index = 0;
    float score = 0.0f;
};
}  // namespace
void non_max_suppression5(const float* boxes_data,
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
    if (soft_nms_sigma > 0.0f) {
        scale = -0.5f / soft_nms_sigma;
    }

    auto func = [iou_threshold, scale](float iou) {
        const float weight = std::exp(scale * iou * iou);
        return iou <= iou_threshold ? weight : 0.0f;
    };

    // boxes shape: {num_batches, num_boxes, 4}
    // scores shape: {num_batches, num_classes, num_boxes}
    int64_t num_batches = static_cast<int64_t>(scores_data_shape[0]);
    int64_t num_classes = static_cast<int64_t>(scores_data_shape[1]);
    int64_t num_boxes = static_cast<int64_t>(boxes_data_shape[1]);

    SelectedIndex* selected_indices_ptr = reinterpret_cast<SelectedIndex*>(selected_indices);
    SelectedScore* selected_scores_ptr = reinterpret_cast<SelectedScore*>(selected_scores);

    size_t boxes_per_class = static_cast<size_t>(max_output_boxes_per_class);

    std::vector<BoxInfo> filteredBoxes;

    for (int64_t batch = 0; batch < num_batches; batch++) {
        const float* boxesPtr = boxes_data + batch * num_boxes * 4;
        Rectangle* r = reinterpret_cast<Rectangle*>(const_cast<float*>(boxesPtr));

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
                    float iou = intersectionOverUnion(next_candidate.box, selected[j].box);
                    next_candidate.score *= func(iou);

                    if (iou >= iou_threshold) {
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
        std::sort(filteredBoxes.begin(), filteredBoxes.end(), [](const BoxInfo& l, const BoxInfo& r) {
            bool is_score_equal = std::fabs(l.score - r.score) < 1e-6;
            return (l.score > r.score) || (is_score_equal && l.batch_index < r.batch_index) ||
                   (is_score_equal && l.batch_index == r.batch_index && l.class_index < r.class_index) ||
                   (is_score_equal && l.batch_index == r.batch_index && l.class_index == r.class_index &&
                    l.index < r.index);
        });
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

    // boxes shape: {num_batches, num_boxes, 4}
    // scores shape: {num_batches, num_classes, num_boxes}
    int64_t num_batches = static_cast<int64_t>(scores_data_shape[0]);
    int64_t num_classes = static_cast<int64_t>(scores_data_shape[1]);
    int64_t num_boxes = static_cast<int64_t>(boxes_data_shape[1]);

    SelectedIndex* selected_indices_ptr = reinterpret_cast<SelectedIndex*>(selected_indices);
    SelectedScore* selected_scores_ptr = reinterpret_cast<SelectedScore*>(selected_scores);

    size_t boxes_per_class = static_cast<size_t>(max_output_boxes_per_class);

    std::vector<BoxInfo> filteredBoxes;

    for (int64_t batch = 0; batch < num_batches; batch++) {
        const float* boxesPtr = boxes_data + batch * num_boxes * 4;
        Rectangle* r = reinterpret_cast<Rectangle*>(const_cast<float*>(boxesPtr));

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
                    float iou = intersectionOverUnion(next_candidate.box, selected[j].box);
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

void nms_postprocessing(ov::TensorVector& outputs,
                        const ov::element::Type output_type,
                        const std::vector<int64_t>& selected_indices,
                        const std::vector<float>& selected_scores,
                        int64_t valid_outputs,
                        const ov::element::Type selected_scores_type) {
    outputs[0].set_shape(Shape{static_cast<size_t>(valid_outputs), 3});

    size_t num_of_outputs = outputs.size();

    if (num_of_outputs >= 2) {
        outputs[1].set_shape(Shape{static_cast<size_t>(valid_outputs), 3});
    }

    if (num_of_outputs >= 3) {
        outputs[2].set_shape(Shape{1});
    }

    size_t selected_size = valid_outputs * 3;

    if (output_type == ov::element::i64) {
        int64_t* indices_ptr = outputs[0].data<int64_t>();
        memcpy(indices_ptr, selected_indices.data(), selected_size * sizeof(int64_t));
    } else {
        int32_t* indices_ptr = outputs[0].data<int32_t>();
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
        bfloat16* scores_ptr = outputs[1].data<bfloat16>();
        for (size_t i = 0; i < selected_scores_size; ++i) {
            scores_ptr[i] = bfloat16(selected_scores[i]);
        }
    } break;
    case element::Type_t::f16: {
        float16* scores_ptr = outputs[1].data<float16>();
        for (size_t i = 0; i < selected_scores_size; ++i) {
            scores_ptr[i] = float16(selected_scores[i]);
        }
    } break;
    case element::Type_t::f32: {
        float* scores_ptr = outputs[1].data<float>();
        memcpy(scores_ptr, selected_scores.data(), selected_size * sizeof(float));
    } break;
    default:;
    }

    if (num_of_outputs < 3) {
        return;
    }

    if (output_type == ov::element::i64) {
        int64_t* valid_outputs_ptr = outputs[2].data<int64_t>();
        *valid_outputs_ptr = valid_outputs;
    } else {
        int32_t* valid_outputs_ptr = outputs[2].data<int32_t>();
        *valid_outputs_ptr = static_cast<int32_t>(valid_outputs);
    }
}

void nms5_postprocessing(ov::TensorVector& outputs,
                         const ov::element::Type output_type,
                         const std::vector<int64_t>& selected_indices,
                         const std::vector<float>& selected_scores,
                         int64_t valid_outputs,
                         const ov::element::Type selected_scores_type) {
    nms_postprocessing(outputs, output_type, selected_indices, selected_scores, valid_outputs, selected_scores_type);
}
}  // namespace reference
}  // namespace ov
