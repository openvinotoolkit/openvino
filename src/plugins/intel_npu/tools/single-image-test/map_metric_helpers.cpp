//
// Copyright (C) 2018-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "map_metric_helpers.hpp"

#include "argument_parse_helpers.hpp"
#include "tensor_utils.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <vector>

namespace utils {

// Matches Python's overlap_evaluator (Overlap class with IOU method)
float calculateIoU(const Detection& detection1, const Detection& detection2, bool include_boundaries) {
    float adjustment = include_boundaries ? 1.0f : 0.0f;

    float x_min_inter = std::max(detection1.x_min, detection2.x_min);
    float y_min_inter = std::max(detection1.y_min, detection2.y_min);
    float x_max_inter = std::min(detection1.x_max, detection2.x_max);
    float y_max_inter = std::min(detection1.y_max, detection2.y_max);

    float inter_width = std::max(0.0f, x_max_inter - x_min_inter + adjustment);
    float inter_height = std::max(0.0f, y_max_inter - y_min_inter + adjustment);
    float intersection = inter_width * inter_height;

    float area1 = (detection1.x_max - detection1.x_min + adjustment) * (detection1.y_max - detection1.y_min + adjustment);
    float area2 = (detection2.x_max - detection2.x_min + adjustment) * (detection2.y_max - detection2.y_min + adjustment);

    float union_area = area1 + area2 - intersection;

    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

// Parse detections from model outputs (single image)
// Expected outputs: pred_boxes, logits, encoder_hidden_state, last_hidden_state
// pred_boxes: [batch, num_queries, 4] with format [x_center, y_center, width, height] (normalized)
// logits: [batch, num_queries, num_classes] with class probabilities/logits
std::vector<Detection> parseDetectionsFromOutputs(const std::map<std::string, ov::Tensor>& outputs,
                                                  float confidence_threshold) {
    std::vector<Detection> detections;

    // Find the pred_boxes and logits tensors
    auto pred_boxes_it = outputs.find("pred_boxes");
    auto logits_it = outputs.find("logits");

    // Parse confidence_threshold (can be per-layer or global)
    auto confMap = parsePerLayerValues(FLAGS_confidence_threshold, metric_defaults::confidence_threshold);
    float confThresh = static_cast<float>(getValueForLayer(confMap, "logits"));
    if (confidence_threshold > 0.0f) {
        confThresh = confidence_threshold;  // Use passed parameter if provided
    }

    if (pred_boxes_it == outputs.end()) {
        std::cout << "Warning: 'pred_boxes' output not found" << std::endl;
        return detections;
    }

    if (logits_it == outputs.end()) {
        std::cout << "Warning: 'logits' output not found" << std::endl;
        return detections;
    }

    const ov::Tensor& pred_boxes_tensor = pred_boxes_it->second;
    const ov::Tensor& logits_tensor = logits_it->second;

    const ov::Tensor boxes_fp32 = npu::utils::toFP32(pred_boxes_tensor);
    const ov::Tensor logits_fp32 = npu::utils::toFP32(logits_tensor);

    const auto boxes_buffer = boxes_fp32.data<const float>();
    const auto logits_buffer = logits_fp32.data<const float>();

    const auto boxes_shape = pred_boxes_tensor.get_shape();
    const auto logits_shape = logits_tensor.get_shape();

    // Expected shapes: pred_boxes [batch, num_queries, 4], logits [batch, num_queries, num_classes]
    if (boxes_shape.size() != 3 || logits_shape.size() != 3) {
        std::cout << "Unexpected tensor shapes - pred_boxes: " << boxes_shape
                  << ", logits: " << logits_shape << std::endl;
        return detections;
    }

    size_t batch_size = boxes_shape[0];
    size_t num_queries = boxes_shape[1];
    size_t box_dim = boxes_shape[2];  // Should be 4
    size_t num_classes = logits_shape[2];

    if (batch_size != 1) {
        std::cout << "Warning: batch_size = " << batch_size << ", expected 1 for single-image inference" << std::endl;
    }

    if (box_dim != 4) {
        std::cout << "Error: Expected 4 box coordinates, got " << box_dim << std::endl;
        return detections;
    }

    if (num_queries != logits_shape[1]) {
        std::cout << "Error: Mismatch between pred_boxes queries (" << num_queries
                  << ") and logits queries (" << logits_shape[1] << ")" << std::endl;
        return detections;
    }

    for (size_t queryIdx = 0; queryIdx < num_queries; ++queryIdx) {
        size_t box_offset = queryIdx * 4;
        float x_center = boxes_buffer[box_offset + 0];
        float y_center = boxes_buffer[box_offset + 1];
        float width = boxes_buffer[box_offset + 2];
        float height = boxes_buffer[box_offset + 3];

        // Convert from [x_center, y_center, w, h] to [x_min, y_min, x_max, y_max]
        float x_min = x_center - width / 2.0f;
        float y_min = y_center - height / 2.0f;
        float x_max = x_center + width / 2.0f;
        float y_max = y_center + height / 2.0f;

        // Get class logits/probabilities
        size_t logits_offset = queryIdx * num_classes;

        // Find class with highest confidence and compute proper softmax
        float max_logit = -std::numeric_limits<float>::infinity();
        int best_class = -1;

        // First pass: find max logit for numerical stability
        for (size_t c = 0; c < num_classes; ++c) {
            float logit = logits_buffer[logits_offset + c];
            if (logit > max_logit) {
                max_logit = logit;
                best_class = static_cast<int>(c);
            }
        }

        // Second pass: compute softmax with numerical stability
        // softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
        float exp_sum = 0.0f;
        for (size_t c = 0; c < num_classes; ++c) {
            float logit = logits_buffer[logits_offset + c];
            exp_sum += std::exp(logit - max_logit);
        }

        // Confidence is the softmax probability of the best class
        float confidence = 1.0f / exp_sum;

        // Filter by confidence threshold
        if (confidence > confidence_threshold && best_class >= 0) {
            detections.emplace_back(x_min, y_min, x_max, y_max, confidence, best_class);
        }
    }

    return detections;
}

MatchResult matchDetectionsForClass(const std::vector<Detection>& predictions,
                                    const std::vector<Detection>& ground_truth,
                                    int class_id,
                                    float iou_threshold,
                                    bool include_boundaries) {
    MatchResult result;

    // Filter predictions and GT for this class
    std::vector<Detection> class_predictions;
    std::vector<Detection> class_gt;

    for (const auto& pred : predictions) {
        if (pred.class_id == class_id) {
            class_predictions.emplace_back(pred);
        }
    }

    for (const auto& gt : ground_truth) {
        if (gt.class_id == class_id) {
            class_gt.emplace_back(gt);
        }
    }

    result.num_ground_truth = class_gt.size();

    if (class_predictions.empty()) {
        return result;
    }

    // Sort predictions by confidence (descending)
    std::sort(class_predictions.begin(), class_predictions.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });

    // Track which GT boxes have been matched
    std::vector<bool> gt_matched(class_gt.size(), false);

    // For each prediction, find best matching GT box
    for (const auto& pred : class_predictions) {
        result.confidences.push_back(pred.confidence);

        float best_iou = 0.0f;
        int best_gt_idx = -1;

        // Find GT box with highest IoU
        for (size_t gt_idx = 0; gt_idx < class_gt.size(); ++gt_idx) {
            if (gt_matched[gt_idx]) {
                continue;  // Already matched
            }

            float iou = calculateIoU(pred, class_gt[gt_idx], include_boundaries);

            if (iou > best_iou) {
                best_iou = iou;
                best_gt_idx = static_cast<int>(gt_idx);
            }
        }

        // Check if match is good enough
        if (best_gt_idx >= 0 && best_iou >= iou_threshold) {
            gt_matched[best_gt_idx] = true;
            result.tp.push_back(1);
            result.fp.push_back(0);
        } else {
            result.tp.push_back(0);
            result.fp.push_back(1);
        }
    }

    return result;
}

// Helper function to calculate Average Precision using VOC max interpolation
// This matches the Python implementation's average_precision() function with APIntegralType.voc_max
double calculateAveragePrecision(const std::vector<float>& precision, const std::vector<float>& recall) {
    if (precision.empty() || recall.empty()) {
        return 0.0;
    }

    // Append sentinel values at the end (matching Python: recall = np.concatenate(([0.], recall, [1.])))
    std::vector<double> recall_with_sentinel;
    std::vector<double> precision_with_sentinel;

    recall_with_sentinel.push_back(0.0);
    precision_with_sentinel.push_back(0.0);

    for (size_t i = 0; i < recall.size(); ++i) {
        recall_with_sentinel.emplace_back(recall[i]);
        precision_with_sentinel.emplace_back(precision[i]);
    }

    recall_with_sentinel.push_back(1.0);
    precision_with_sentinel.push_back(0.0);

    // Compute the precision envelope (make precision monotonically decreasing)
    // Python: for i in range(precision.size - 1, 0, -1): precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    for (int i = static_cast<int>(precision_with_sentinel.size()) - 1; i > 0; --i) {
        precision_with_sentinel[i - 1] = std::max(precision_with_sentinel[i - 1], precision_with_sentinel[i]);
    }

    // Find points where X axis (recall) changes value
    // Python: change_point = np.where(recall[1:] != recall[:-1])[0]
    std::vector<size_t> change_points;
    for (size_t i = 0; i < recall_with_sentinel.size() - 1; ++i) {
        if (recall_with_sentinel[i + 1] != recall_with_sentinel[i]) {
            change_points.push_back(i);
        }
    }

    // Sum (\Delta recall) * precision
    // Python: np.sum((recall[change_point + 1] - recall[change_point]) * precision[change_point + 1])
    double ap = 0.0;
    for (size_t cp : change_points) {
        ap += (recall_with_sentinel[cp + 1] - recall_with_sentinel[cp]) * precision_with_sentinel[cp + 1];
    }

    return ap;
}

}  // namespace utils
