// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/matrix_nms.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/reference/matrix_nms.hpp"
#include "openvino/reference/utils/nms_common.hpp"

namespace ov {
namespace reference {
namespace matrix_nms_v8 {
template <typename T, bool gaussian>
struct decay_score;

template <typename T>
struct decay_score<T, true> {
    T operator()(T iou, T max_iou, T sigma) {
        return std::exp((max_iou * max_iou - iou * iou) * sigma);
    }
};

template <typename T>
struct decay_score<T, false> {
    T operator()(T iou, T max_iou, T sigma) {
        return static_cast<T>((1. - iou) / (1. - max_iou + 1e-10f));
    }
};

template <class T>
static inline T BBoxArea(const T* box, const bool normalized) {
    if (box[2] < box[0] || box[3] < box[1]) {
        // If coordinate values are is invalid
        // (e.g. xmax < xmin or ymax < ymin), return 0.
        return static_cast<T>(0.);
    } else {
        const T w = box[2] - box[0];
        const T h = box[3] - box[1];
        if (normalized) {
            return w * h;
        } else {
            // If coordinate values are not within range [0, 1].
            return (w + 1) * (h + 1);
        }
    }
}

template <class T>
static inline T intersectionOverUnion(const T* box1, const T* box2, const bool normalized) {
    if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] || box2[3] < box1[1]) {
        return static_cast<T>(0.);
    } else {
        const T inter_xmin = std::max(box1[0], box2[0]);
        const T inter_ymin = std::max(box1[1], box2[1]);
        const T inter_xmax = std::min(box1[2], box2[2]);
        const T inter_ymax = std::min(box1[3], box2[3]);
        T norm = normalized ? static_cast<T>(0.) : static_cast<T>(1.);
        T inter_w = inter_xmax - inter_xmin + norm;
        T inter_h = inter_ymax - inter_ymin + norm;
        const T inter_area = inter_w * inter_h;
        const T bbox1_area = BBoxArea<T>(box1, normalized);
        const T bbox2_area = BBoxArea<T>(box2, normalized);
        return inter_area / (bbox1_area + bbox2_area - inter_area);
    }
}
}  // namespace matrix_nms_v8

template <typename T, bool gaussian>
void nms_matrix(const T* boxes_data,
                const Shape& boxes_data_shape,
                const T* scores_data,
                const Shape& scores_data_shape,
                const T score_threshold,
                const T post_threshold,
                const float sigma,
                const int64_t top_k,
                const bool normalized,
                std::vector<int>* selected_indices,
                std::vector<T>* decayed_scores) {
    int64_t boxes_num = static_cast<int64_t>(boxes_data_shape[1]);
    int64_t box_size = static_cast<int64_t>(boxes_data_shape[2]);

    std::vector<int32_t> candidate_index(boxes_num);
    std::iota(candidate_index.begin(), candidate_index.end(), 0);
    auto end =
        std::remove_if(candidate_index.begin(), candidate_index.end(), [&scores_data, score_threshold](int32_t idx) {
            return scores_data[idx] <= score_threshold;
        });

    int64_t original_size = std::distance(candidate_index.begin(), end);
    if (original_size <= 0) {
        return;
    }
    if (top_k > -1 && original_size > top_k) {
        original_size = top_k;
    }

    std::partial_sort(candidate_index.begin(),
                      candidate_index.begin() + original_size,
                      end,
                      [&scores_data](int32_t a, int32_t b) {
                          return scores_data[a] > scores_data[b];
                      });

    std::vector<T> iou_matrix((original_size * (original_size - 1)) >> 1);
    std::vector<T> iou_max(original_size);

    iou_max[0] = 0.;
    for (int64_t i = 1; i < original_size; i++) {
        T max_iou = 0.;
        auto idx_a = candidate_index[i];
        for (int64_t j = 0; j < i; j++) {
            auto idx_b = candidate_index[j];
            auto iou = matrix_nms_v8::intersectionOverUnion<T>(boxes_data + idx_a * box_size,
                                                               boxes_data + idx_b * box_size,
                                                               normalized);
            max_iou = std::max(max_iou, iou);
            iou_matrix[i * (i - 1) / 2 + j] = iou;
        }
        iou_max[i] = max_iou;
    }

    if (scores_data[candidate_index[0]] > post_threshold) {
        selected_indices->push_back(candidate_index[0]);
        decayed_scores->push_back(scores_data[candidate_index[0]]);
    }

    matrix_nms_v8::decay_score<T, gaussian> decay_fn;
    for (int64_t i = 1; i < original_size; i++) {
        T min_decay = 1.;
        for (int64_t j = 0; j < i; j++) {
            auto max_iou = iou_max[j];
            auto iou = iou_matrix[i * (i - 1) / 2 + j];
            auto decay = decay_fn(iou, max_iou, sigma);
            min_decay = std::min(min_decay, decay);
        }
        auto ds = min_decay * scores_data[candidate_index[i]];
        if (ds <= post_threshold)
            continue;
        selected_indices->push_back(candidate_index[i]);
        decayed_scores->push_back(ds);
    }
}

void matrix_nms(const float* boxes_data,
                const Shape& boxes_data_shape,
                const float* scores_data,
                const Shape& scores_data_shape,
                const op::v8::MatrixNms::Attributes& attrs,
                float* selected_outputs,
                const Shape& selected_outputs_shape,
                int64_t* selected_indices,
                const Shape& selected_indices_shape,
                int64_t* valid_outputs) {
    using Rectangle = reference::nms_common::Rectangle;
    using BoxInfo = reference::nms_common::BoxInfo;

    // boxes shape: {num_batches, num_boxes, 4}
    // scores shape: {num_batches, num_classes, num_boxes}
    int64_t num_batches = static_cast<int64_t>(scores_data_shape[0]);
    int64_t num_classes = static_cast<int64_t>(scores_data_shape[1]);
    int64_t num_boxes = static_cast<int64_t>(boxes_data_shape[1]);
    int64_t box_shape = static_cast<int64_t>(boxes_data_shape[2]);

    std::vector<int> num_per_batch;
    std::vector<BoxInfo> filtered_boxes;
    filtered_boxes.reserve(6 * num_batches * num_classes * num_boxes);

    for (int64_t batch = 0; batch < num_batches; batch++) {
        const float* boxesPtr = boxes_data + batch * num_boxes * 4;
        std::vector<int> all_indices;
        std::vector<float> all_scores;
        std::vector<int64_t> all_classes;
        size_t num_det = 0;

        for (int64_t class_idx = 0; class_idx < num_classes; class_idx++) {
            if (class_idx == attrs.background_class)
                continue;
            const float* scoresPtr = scores_data + batch * (num_classes * num_boxes) + class_idx * num_boxes;
            if (attrs.decay_function == op::v8::MatrixNms::DecayFunction::GAUSSIAN) {
                nms_matrix<float, true>(boxesPtr,
                                        boxes_data_shape,
                                        scoresPtr,
                                        scores_data_shape,
                                        attrs.score_threshold,
                                        attrs.post_threshold,
                                        attrs.gaussian_sigma,
                                        attrs.nms_top_k,
                                        attrs.normalized,
                                        &all_indices,
                                        &all_scores);
            } else {
                nms_matrix<float, false>(boxesPtr,
                                         boxes_data_shape,
                                         scoresPtr,
                                         scores_data_shape,
                                         attrs.score_threshold,
                                         attrs.post_threshold,
                                         attrs.gaussian_sigma,
                                         attrs.nms_top_k,
                                         attrs.normalized,
                                         &all_indices,
                                         &all_scores);
            }
            for (size_t i = 0; i < all_indices.size() - num_det; i++) {
                all_classes.push_back(class_idx);
            }
            num_det = all_indices.size();
        }

        if (num_det <= 0) {
            break;
        }

        if (attrs.keep_top_k > -1) {
            auto k = static_cast<size_t>(attrs.keep_top_k);
            if (num_det > k)
                num_det = k;
        }

        std::vector<int32_t> perm(all_indices.size());
        std::iota(perm.begin(), perm.end(), 0);

        std::partial_sort(perm.begin(),
                          perm.begin() + num_det,
                          perm.end(),
                          [&all_scores, &all_classes, &all_indices](int lhs, int rhs) {
                              return (all_scores[lhs] > all_scores[rhs]) ||
                                     (all_scores[lhs] == all_scores[rhs] && all_classes[lhs] < all_classes[rhs]) ||
                                     (all_scores[lhs] == all_scores[rhs] && all_classes[lhs] == all_classes[rhs] &&
                                      all_indices[lhs] < all_indices[rhs]);
                          });

        for (size_t i = 0; i < num_det; i++) {
            auto p = perm[i];
            auto idx = all_indices[p];
            auto cls = all_classes[p];
            auto score = all_scores[p];
            auto bbox = boxesPtr + idx * box_shape;

            filtered_boxes.push_back(
                BoxInfo{Rectangle{bbox[0], bbox[1], bbox[2], bbox[3]}, batch * num_boxes + idx, score, 0, batch, cls});
        }
        num_per_batch.push_back(static_cast<int32_t>(num_det));
    }

    if (attrs.sort_result_across_batch) { /* sort across batch */
        if (attrs.sort_result_type == op::v8::MatrixNms::SortResultType::SCORE) {
            std::sort(filtered_boxes.begin(), filtered_boxes.end(), [](const BoxInfo& l, const BoxInfo& r) {
                return (l.score > r.score) || (l.score == r.score && l.batch_index < r.batch_index) ||
                       (l.score == r.score && l.batch_index == r.batch_index && l.class_index < r.class_index) ||
                       (l.score == r.score && l.batch_index == r.batch_index && l.class_index == r.class_index &&
                        l.index < r.index);
            });
        } else if (attrs.sort_result_type == op::v8::MatrixNms::SortResultType::CLASSID) {
            std::sort(filtered_boxes.begin(), filtered_boxes.end(), [](const BoxInfo& l, const BoxInfo& r) {
                return (l.class_index < r.class_index) ||
                       (l.class_index == r.class_index && l.batch_index < r.batch_index) ||
                       (l.class_index == r.class_index && l.batch_index == r.batch_index && l.score > r.score) ||
                       (l.class_index == r.class_index && l.batch_index == r.batch_index && l.score == r.score &&
                        l.index < r.index);
            });
        }
    }

    std::copy(num_per_batch.begin(), num_per_batch.end(), valid_outputs);
    for (size_t i = 0; i < filtered_boxes.size(); i++) {
        selected_indices[i] = filtered_boxes[i].index;
        auto selected_base = selected_outputs + i * 6;
        selected_base[0] = static_cast<float>(filtered_boxes[i].class_index);
        selected_base[1] = filtered_boxes[i].score;
        selected_base[2] = filtered_boxes[i].box.x1;
        selected_base[3] = filtered_boxes[i].box.y1;
        selected_base[4] = filtered_boxes[i].box.x2;
        selected_base[5] = filtered_boxes[i].box.y2;
    }
}
}  // namespace reference
}  // namespace ov
