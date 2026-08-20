// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include <openvino/runtime/tensor.hpp>

namespace utils {

// Structure to represent a detection bounding box
// For single-image inference, image_id is always 0
struct Detection {
    float x_min;
    float y_min;
    float x_max;
    float y_max;
    float confidence;
    int class_id;

    Detection(float xmin, float ymin, float xmax, float ymax, float conf, int cls)
        : x_min(xmin), y_min(ymin), x_max(xmax), y_max(ymax), confidence(conf), class_id(cls) {}
};

// Match predictions to ground truth boxes for a specific class
// Implements Python's bbox_match() function
struct MatchResult {
    std::vector<int> tp;            // True positives
    std::vector<int> fp;            // False positives
    std::vector<float> confidences; // Confidence scores
    size_t num_ground_truth;        // Total number of GT boxes for this class
};

double calculateAveragePrecision(const std::vector<float>& precision, const std::vector<float>& recall);

MatchResult matchDetectionsForClass(const std::vector<Detection>& predictions,
                                    const std::vector<Detection>& ground_truth,
                                    int class_id,
                                    float iou_threshold,
                                    bool include_boundaries = true);

// Parse detections from model outputs.
// Supports multiple output formats:
//   - Two-layer mode: outputs contain "pred_boxes" [batch, N, 4] (cx, cy, w, h normalized)
//     and "logits" [batch, N, num_classes]
//   - Single-layer mode (e.g. YOLOv10): single output with shape [batch, N, 6]
//     where each detection is [x1, y1, x2, y2, score, class_id]
//   - Single-layer mode with per-class scores: single output with shape [batch, N, 5+C]
//     where each detection is [x1, y1, x2, y2, score, class_0_score, ..., class_C-1_score]
std::vector<Detection> parseDetectionsFromOutputs(const std::map<std::string, ov::Tensor>& outputs,
                                     float confidence_threshold = 0.0f);
}  // namespace utils
