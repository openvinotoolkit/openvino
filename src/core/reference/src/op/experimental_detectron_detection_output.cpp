//*****************************************************************************
// Copyright 2017-2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "openvino/op/experimental_detectron_detection_output.hpp"

#include <algorithm>
#include <cassert>
#include <utility>

#include "openvino/core/shape.hpp"
#include "openvino/reference/experimental_detectron_detection_output.hpp"

namespace {
void refine_boxes(const float* boxes,
                  const float* deltas,
                  const float* weights,
                  const float* scores,
                  float* refined_boxes,
                  float* refined_boxes_areas,
                  float* refined_scores,
                  const int64_t rois_num,
                  const int64_t classes_num,
                  const float img_H,
                  const float img_W,
                  const float max_delta_log_wh,
                  float coordinates_offset) {
    for (int64_t roi_idx = 0; roi_idx < rois_num; ++roi_idx) {
        float x0 = boxes[roi_idx * 4 + 0];
        float y0 = boxes[roi_idx * 4 + 1];
        float x1 = boxes[roi_idx * 4 + 2];
        float y1 = boxes[roi_idx * 4 + 3];

        if (x1 - x0 <= 0 || y1 - y0 <= 0) {
            continue;
        }

        // width & height of box
        const float ww = x1 - x0 + coordinates_offset;
        const float hh = y1 - y0 + coordinates_offset;
        // center location of box
        const float ctr_x = x0 + 0.5f * ww;
        const float ctr_y = y0 + 0.5f * hh;

        for (int class_idx = 1; class_idx < classes_num; ++class_idx) {
            const int64_t deltas_base_offset = classes_num * 4 * roi_idx + 4 * class_idx;
            const float dx = deltas[deltas_base_offset + 0] / weights[0];
            const float dy = deltas[deltas_base_offset + 1] / weights[1];
            const float d_log_w = deltas[deltas_base_offset + 2] / weights[2];
            const float d_log_h = deltas[deltas_base_offset + 3] / weights[3];

            // new center location according to deltas (dx, dy)
            const float pred_ctr_x = dx * ww + ctr_x;
            const float pred_ctr_y = dy * hh + ctr_y;
            // new width & height according to deltas d(log w), d(log h)
            const float pred_w = std::exp(std::min(d_log_w, max_delta_log_wh)) * ww;
            const float pred_h = std::exp(std::min(d_log_h, max_delta_log_wh)) * hh;

            // update upper-left corner location
            float x0_new = pred_ctr_x - 0.5f * pred_w;
            float y0_new = pred_ctr_y - 0.5f * pred_h;
            // update lower-right corner location
            float x1_new = pred_ctr_x + 0.5f * pred_w - coordinates_offset;
            float y1_new = pred_ctr_y + 0.5f * pred_h - coordinates_offset;

            // adjust new corner locations to be within the image region,
            x0_new = std::max<float>(0.0f, x0_new);
            y0_new = std::max<float>(0.0f, y0_new);
            x1_new = std::max<float>(0.0f, x1_new);
            y1_new = std::max<float>(0.0f, y1_new);

            // recompute new width & height
            const float box_w = x1_new - x0_new + coordinates_offset;
            const float box_h = y1_new - y0_new + coordinates_offset;

            const int64_t refined_boxes_base_offset = rois_num * 4 * class_idx + 4 * roi_idx;
            refined_boxes[refined_boxes_base_offset + 0] = x0_new;
            refined_boxes[refined_boxes_base_offset + 1] = y0_new;
            refined_boxes[refined_boxes_base_offset + 2] = x1_new;
            refined_boxes[refined_boxes_base_offset + 3] = y1_new;

            const int64_t refined_score_offset = rois_num * class_idx + roi_idx;
            const int64_t scores_offset = classes_num * roi_idx + class_idx;
            refined_boxes_areas[refined_score_offset] = box_w * box_h;
            refined_scores[refined_score_offset] = scores[scores_offset];
        }
    }
}

struct ConfidenceComparator {
    explicit ConfidenceComparator(const float* conf_data) : m_conf_data(conf_data) {}

    bool operator()(int64_t idx1, int64_t idx2) {
        if (m_conf_data[idx1] > m_conf_data[idx2])
            return true;
        if (m_conf_data[idx1] < m_conf_data[idx2])
            return false;
        return idx1 < idx2;
    }

    const float* m_conf_data;
};

inline float JaccardOverlap(const float* decoded_bbox,
                            const float* bbox_sizes,
                            const int64_t idx1,
                            const int64_t idx2,
                            const float coordinates_offset = 1) {
    float xmin1 = decoded_bbox[idx1 * 4 + 0];
    float ymin1 = decoded_bbox[idx1 * 4 + 1];
    float xmax1 = decoded_bbox[idx1 * 4 + 2];
    float ymax1 = decoded_bbox[idx1 * 4 + 3];

    float xmin2 = decoded_bbox[idx2 * 4 + 0];
    float ymin2 = decoded_bbox[idx2 * 4 + 1];
    float ymax2 = decoded_bbox[idx2 * 4 + 3];
    float xmax2 = decoded_bbox[idx2 * 4 + 2];

    const bool bbox_not_covered = xmin2 > xmax1 || xmax2 < xmin1 || ymin2 > ymax1 || ymax2 < ymin1;
    if (bbox_not_covered) {
        return 0.0f;
    }

    float intersect_xmin = std::max(xmin1, xmin2);
    float intersect_ymin = std::max(ymin1, ymin2);
    float intersect_xmax = std::min(xmax1, xmax2);
    float intersect_ymax = std::min(ymax1, ymax2);

    float intersect_width = intersect_xmax - intersect_xmin + coordinates_offset;
    float intersect_height = intersect_ymax - intersect_ymin + coordinates_offset;

    if (intersect_width <= 0 || intersect_height <= 0) {
        return 0.0f;
    }

    float intersect_size = intersect_width * intersect_height;
    float bbox1_size = bbox_sizes[idx1];
    float bbox2_size = bbox_sizes[idx2];

    return intersect_size / (bbox1_size + bbox2_size - intersect_size);
}

void nms_cf(const float* conf_data,
            const float* bboxes,
            const float* sizes,
            int64_t* buffer,
            int64_t* indices,
            int64_t& detections,
            const int64_t boxes_num,
            const int64_t pre_nms_topn,
            const int64_t post_nms_topn,
            const float confidence_threshold,
            const float nms_threshold) {
    int64_t count = 0;
    for (int64_t i = 0; i < boxes_num; ++i) {
        if (conf_data[i] > confidence_threshold) {
            indices[count] = i;
            count++;
        }
    }

    int64_t num_output_scores = (pre_nms_topn == -1 ? count : (std::min)(pre_nms_topn, count));

    std::partial_sort_copy(indices,
                           indices + count,
                           buffer,
                           buffer + num_output_scores,
                           ConfidenceComparator(conf_data));

    detections = 0;
    for (int64_t i = 0; i < num_output_scores; ++i) {
        const int64_t idx = buffer[i];

        bool keep = true;
        for (int64_t k = 0; k < detections; ++k) {
            const int64_t kept_idx = indices[k];
            float overlap = JaccardOverlap(bboxes, sizes, idx, kept_idx);
            if (overlap > nms_threshold) {
                keep = false;
                break;
            }
        }
        if (keep) {
            indices[detections] = idx;
            detections++;
        }
    }

    detections = (post_nms_topn == -1 ? detections : (std::min)(post_nms_topn, detections));
}

template <typename T>
bool SortScorePairDescend(const std::pair<float, T>& pair1, const std::pair<float, T>& pair2) {
    return (pair1.first > pair2.first) || ((pair1.first == pair2.first) && (pair1.second.second < pair2.second.second));
}
}  // namespace

namespace ov {
namespace reference {
void experimental_detectron_detection_output(const float* boxes,
                                             const float* input_deltas,
                                             const float* input_scores,
                                             const float* input_im_info,
                                             const op::v6::ExperimentalDetectronDetectionOutput::Attributes& attrs,
                                             float* output_boxes,
                                             float* output_scores,
                                             int32_t* output_classes) {
    const float img_H = input_im_info[0];
    const float img_W = input_im_info[1];
    const int64_t classes_num = attrs.num_classes;
    const int64_t rois_num = static_cast<int64_t>(attrs.max_detections_per_image);
    const int64_t max_detections_per_image = static_cast<int64_t>(attrs.max_detections_per_image);
    const int64_t max_detections_per_class = attrs.post_nms_count;
    const float score_threshold = attrs.score_threshold;
    const float nms_threshold = attrs.nms_threshold;

    const auto& deltas_weights = attrs.deltas_weights;
    const float max_delta_log_wh = attrs.max_delta_log_wh;

    // Apply deltas.
    std::vector<float> refined_boxes(classes_num * rois_num * 4, 0);
    std::vector<float> refined_scores(classes_num * rois_num, 0);
    std::vector<float> refined_boxes_areas(classes_num * rois_num, 0);

    refine_boxes(boxes,
                 input_deltas,
                 deltas_weights.data(),
                 input_scores,
                 refined_boxes.data(),
                 refined_boxes_areas.data(),
                 refined_scores.data(),
                 rois_num,
                 classes_num,
                 img_H,
                 img_W,
                 max_delta_log_wh,
                 1.0f);

    // Apply NMS class-wise.
    std::vector<int64_t> buffer(rois_num, 0);
    std::vector<int64_t> indices(classes_num * rois_num, 0);
    std::vector<int64_t> detections_per_class(classes_num, 0);
    int64_t total_detections_num = 0;

    for (int64_t class_idx = 1; class_idx < classes_num; ++class_idx) {
        nms_cf(&refined_scores[rois_num * class_idx],
               &refined_boxes[rois_num * 4 * class_idx],
               &refined_boxes_areas[rois_num * class_idx],
               &buffer[0],
               &indices[total_detections_num],
               detections_per_class[class_idx],
               rois_num,
               -1,
               max_detections_per_class,
               score_threshold,
               nms_threshold);
        total_detections_num += detections_per_class[class_idx];
    }

    // Leave only max_detections_per_image detections.
    // confidence, <class, index>
    std::vector<std::pair<float, std::pair<int64_t, int64_t>>> conf_index_class_map;

    int64_t indices_offset = 0;
    for (int64_t c = 0; c < classes_num; ++c) {
        int64_t n = detections_per_class[c];
        for (int64_t i = 0; i < n; ++i) {
            int64_t idx = indices[indices_offset + i];
            float score = refined_scores[rois_num * c + idx];
            conf_index_class_map.emplace_back(score, std::make_pair(c, idx));
        }
        indices_offset += n;
    }

    assert(max_detections_per_image > 0);
    if (total_detections_num > max_detections_per_image) {
        std::partial_sort(conf_index_class_map.begin(),
                          conf_index_class_map.begin() + max_detections_per_image,
                          conf_index_class_map.end(),
                          SortScorePairDescend<std::pair<int64_t, int64_t>>);
        conf_index_class_map.resize(max_detections_per_image);
        total_detections_num = max_detections_per_image;
    }

    // Fill outputs.
    memset(output_boxes, 0, max_detections_per_image * 4 * sizeof(output_boxes[0]));
    memset(output_scores, 0, max_detections_per_image * sizeof(output_scores[0]));
    memset(output_classes, 0, max_detections_per_image * sizeof(output_classes[0]));

    int64_t i = 0;
    for (const auto& detection : conf_index_class_map) {
        float score = detection.first;
        int64_t cls = detection.second.first;
        int64_t idx = detection.second.second;
        int64_t refine_boxes_base_offset = rois_num * 4 * cls + 4 * idx;
        output_boxes[4 * i + 0] = refined_boxes[refine_boxes_base_offset + 0];
        output_boxes[4 * i + 1] = refined_boxes[refine_boxes_base_offset + 1];
        output_boxes[4 * i + 2] = refined_boxes[refine_boxes_base_offset + 2];
        output_boxes[4 * i + 3] = refined_boxes[refine_boxes_base_offset + 3];
        output_scores[i] = score;
        output_classes[i] = static_cast<int32_t>(cls);
        ++i;
    }
}

void experimental_detectron_detection_output_postprocessing(void* pboxes,
                                                            void* pclasses,
                                                            void* pscores,
                                                            const element::Type output_type,
                                                            const std::vector<float>& output_boxes,
                                                            const std::vector<int32_t>& output_classes,
                                                            const std::vector<float>& output_scores,
                                                            const Shape& output_boxes_shape,
                                                            const Shape& output_classes_shape,
                                                            const Shape& output_scores_shape) {
    size_t rois_num = output_boxes_shape[0];

    switch (output_type) {
    case element::Type_t::bf16: {
        bfloat16* boxes_ptr = reinterpret_cast<bfloat16*>(pboxes);
        bfloat16* scores_ptr = reinterpret_cast<bfloat16*>(pscores);
        for (size_t i = 0; i < rois_num; ++i) {
            boxes_ptr[4 * i + 0] = bfloat16(output_boxes[4 * i + 0]);
            boxes_ptr[4 * i + 1] = bfloat16(output_boxes[4 * i + 1]);
            boxes_ptr[4 * i + 2] = bfloat16(output_boxes[4 * i + 2]);
            boxes_ptr[4 * i + 3] = bfloat16(output_boxes[4 * i + 3]);
            scores_ptr[i] = bfloat16(output_scores[i]);
        }
    } break;
    case element::Type_t::f16: {
        float16* boxes_ptr = reinterpret_cast<float16*>(pboxes);
        float16* scores_ptr = reinterpret_cast<float16*>(pscores);
        for (size_t i = 0; i < rois_num; ++i) {
            boxes_ptr[4 * i + 0] = float16(output_boxes[4 * i + 0]);
            boxes_ptr[4 * i + 1] = float16(output_boxes[4 * i + 1]);
            boxes_ptr[4 * i + 2] = float16(output_boxes[4 * i + 2]);
            boxes_ptr[4 * i + 3] = float16(output_boxes[4 * i + 3]);
            scores_ptr[i] = float16(output_scores[i]);
        }
    } break;
    case element::Type_t::f32: {
        float* boxes_ptr = reinterpret_cast<float*>(pboxes);
        float* scores_ptr = reinterpret_cast<float*>(pscores);
        memcpy(boxes_ptr, output_boxes.data(), shape_size(output_boxes_shape) * sizeof(float));
        memcpy(scores_ptr, output_scores.data(), shape_size(output_scores_shape) * sizeof(float));
    } break;
    default:;
    }

    int32_t* classes_ptr = reinterpret_cast<int32_t*>(pclasses);
    memcpy(classes_ptr, output_classes.data(), shape_size(output_classes_shape) * sizeof(int32_t));
}
}  // namespace reference
}  // namespace ov
