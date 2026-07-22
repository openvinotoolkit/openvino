// Copyright (C) 2018-2026 Intel Corporation
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

// Parse detections from a single combined output tensor.
// Supports two formats:
//   - [batch, N, 6]: each detection is [x1, y1, x2, y2, score, class_id] (e.g. YOLOv10)
//   - [batch, N, 5+C]: each detection is [x1, y1, x2, y2, score, class_0, ..., class_C-1]
//     where score is objectness and class scores are per-class confidences
static std::vector<Detection> parseSingleTensorDetections(const ov::Tensor& tensor,
                                                          float confidence_threshold) {
    std::vector<Detection> detections;

    const ov::Tensor fp32 = npu::utils::toFP32(tensor);
    const auto buffer = fp32.data<const float>();
    const auto shape = tensor.get_shape();

    if (shape.size() != 3) {
        std::cout << "Warning: Single-tensor detection requires 3D shape [batch, N, cols], got "
                  << shape.size() << "D" << std::endl;
        return detections;
    }

    size_t num_detections = shape[1];
    size_t cols = shape[2];

    if (cols < 6) {
        std::cout << "Warning: Single-tensor detection requires at least 6 columns, got "
                  << cols << std::endl;
        return detections;
    }

    if (cols == 6) {
        // YOLOv10 format: [x1, y1, x2, y2, score, class_id]
        for (size_t i = 0; i < num_detections; ++i) {
            size_t offset = i * cols;
            float x1 = buffer[offset + 0];
            float y1 = buffer[offset + 1];
            float x2 = buffer[offset + 2];
            float y2 = buffer[offset + 3];
            float score = buffer[offset + 4];
            int class_id = static_cast<int>(buffer[offset + 5]);

            // Skip padding detections (score <= 0 or invalid boxes)
            if (score > confidence_threshold && x2 > x1 && y2 > y1) {
                detections.emplace_back(x1, y1, x2, y2, score, class_id);
            }
        }
    } else {
        // Format [x1, y1, x2, y2, objectness_score, class_0_score, ..., class_C-1_score]
        size_t num_classes = cols - 5;
        for (size_t i = 0; i < num_detections; ++i) {
            size_t offset = i * cols;
            float x1 = buffer[offset + 0];
            float y1 = buffer[offset + 1];
            float x2 = buffer[offset + 2];
            float y2 = buffer[offset + 3];
            float obj_score = buffer[offset + 4];

            if (obj_score <= confidence_threshold || x2 <= x1 || y2 <= y1) {
                continue;
            }

            // Find the class with the highest score
            float max_class_score = -std::numeric_limits<float>::infinity();
            int best_class = 0;
            for (size_t c = 0; c < num_classes; ++c) {
                float class_score = buffer[offset + 5 + c];
                if (class_score > max_class_score) {
                    max_class_score = class_score;
                    best_class = static_cast<int>(c);
                }
            }

            float confidence = obj_score * max_class_score;
            if (confidence > confidence_threshold) {
                detections.emplace_back(x1, y1, x2, y2, confidence, best_class);
            }
        }
    }

    return detections;
}

// Parse detections from two separate output tensors (pred_boxes + logits).
// pred_boxes: [batch, num_queries, 4] with format [x_center, y_center, width, height] (normalized)
// logits: [batch, num_queries, num_classes] with class probabilities/logits
static std::vector<Detection> parseTwoTensorDetections(const ov::Tensor& pred_boxes_tensor,
                                                       const ov::Tensor& logits_tensor,
                                                       float confidence_threshold) {
    std::vector<Detection> detections;

    const ov::Tensor boxes_fp32 = npu::utils::toFP32(pred_boxes_tensor);
    const ov::Tensor logits_fp32 = npu::utils::toFP32(logits_tensor);

    const auto boxes_buffer = boxes_fp32.data<const float>();
    const auto logits_buffer = logits_fp32.data<const float>();

    const auto boxes_shape = pred_boxes_tensor.get_shape();
    const auto logits_shape = logits_tensor.get_shape();

    if (boxes_shape.size() != 3 || logits_shape.size() != 3) {
        std::cout << "Unexpected tensor shapes - pred_boxes: " << boxes_shape
                  << ", logits: " << logits_shape << std::endl;
        return detections;
    }

    size_t num_queries = boxes_shape[1];
    size_t box_dim = boxes_shape[2];
    size_t num_classes = logits_shape[2];

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

        float x_min = x_center - width / 2.0f;
        float y_min = y_center - height / 2.0f;
        float x_max = x_center + width / 2.0f;
        float y_max = y_center + height / 2.0f;

        size_t logits_offset = queryIdx * num_classes;

        float max_logit = -std::numeric_limits<float>::infinity();
        int best_class = -1;

        for (size_t c = 0; c < num_classes; ++c) {
            float logit = logits_buffer[logits_offset + c];
            if (logit > max_logit) {
                max_logit = logit;
                best_class = static_cast<int>(c);
            }
        }

        float exp_sum = 0.0f;
        for (size_t c = 0; c < num_classes; ++c) {
            float logit = logits_buffer[logits_offset + c];
            exp_sum += std::exp(logit - max_logit);
        }

        float confidence = 1.0f / exp_sum;

        if (confidence > confidence_threshold && best_class >= 0) {
            detections.emplace_back(x_min, y_min, x_max, y_max, confidence, best_class);
        }
    }

    return detections;
}

// Automatically detects the output formats for DETR and YOLOv10 style models
std::vector<Detection> parseDetectionsFromOutputs(const std::map<std::string, ov::Tensor>& outputs,
                                                  float confidence_threshold) {
    std::vector<Detection> detections;

    if (outputs.empty()) {
        std::cout << "Warning: No output tensors provided" << std::endl;
        return detections;
    }

    // Strategy 1: Look for named "pred_boxes" and "logits" tensors (DETR-style)
    auto pred_boxes_it = outputs.find("pred_boxes");
    auto logits_it = outputs.find("logits");

    if (pred_boxes_it != outputs.end() && logits_it != outputs.end()) {
        return parseTwoTensorDetections(pred_boxes_it->second, logits_it->second, confidence_threshold);
    }

    // Strategy 2: Single output tensor so parse as combined detections
    if (outputs.size() == 1) {
        const auto& [name, tensor] = *outputs.begin();
        const auto shape = tensor.get_shape();

        if (shape.size() == 3 && shape[2] >= 6) {
            return parseSingleTensorDetections(tensor, confidence_threshold);
        }

        std::cout << "Warning: Single output '" << name << "' with shape " << shape
                  << " does not match expected detection format [batch, N, 6+]" << std::endl;
        return detections;
    }

    // Strategy 3: Two outputs without standard names so infer roles by shape
    // The tensor with last dim == 4 is boxes, the other is logits/classes
    if (outputs.size() == 2) {
        const ov::Tensor* boxes_tensor = nullptr;
        const ov::Tensor* classes_tensor = nullptr;
        std::string boxes_name, classes_name;

        for (const auto& [name, tensor] : outputs) {
            const auto shape = tensor.get_shape();
            if (shape.size() == 3 && shape[2] == 4) {
                boxes_tensor = &tensor;
                boxes_name = name;
            } else {
                classes_tensor = &tensor;
                classes_name = name;
            }
        }

        if (boxes_tensor && classes_tensor) {
            return parseTwoTensorDetections(*boxes_tensor, *classes_tensor, confidence_threshold);
        }
    }

    std::cout << "Warning: Could not determine detection format from outputs. Available tensors:" << std::endl;
    for (const auto& [name, tensor] : outputs) {
        std::cout << "  " << name << " : " << tensor.get_shape() << std::endl;
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
