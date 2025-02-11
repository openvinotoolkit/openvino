//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <openvino/core/layout.hpp>
#include <openvino/runtime/tensor.hpp>

#include <sstream>

namespace utils {
struct Box final {
    float x, y, w, h;
};

struct BoundingBox final {
    int idx;
    float left, right, top, bottom;
    float prob;
    BoundingBox(int idx, float xmin, float ymin, float xmax, float ymax, float prob)
            : idx(idx), left(xmin), right(xmax), top(ymin), bottom(ymax), prob(prob) {
    }
};

std::vector<BoundingBox> parseYoloOutput(const ov::Tensor& tensor, const size_t imgWidth, const size_t imgHeight,
                                         const float confThresh, const bool isTiny);

std::vector<BoundingBox> parseYoloV3Output(const std::map<std::string, ov::Tensor>& tensors, const size_t imgWidth,
                                           const size_t imgHeight, const int classes, const int coords, const int num,
                                           const std::vector<float>& anchors, const float confThresh,
                                           const std::unordered_map<std::string, ov::Layout>& layouts);

std::vector<BoundingBox> parseYoloV4Output(const std::map<std::string, ov::Tensor>& tensors, const size_t imgWidth,
                                           const size_t imgHeight, const int classes, const int coords, const int num,
                                           const std::vector<float>& anchors, const float confThresh,
                                           const std::unordered_map<std::string, ov::Layout>& layouts);

std::vector<BoundingBox> parseYoloV3V4Output(const std::map<std::string, ov::Tensor>& tensors, const size_t imgWidth,
                                             const size_t imgHeight, const int classes, const int coords, const int num,
                                             const std::vector<float>& anchors, const float confThresh,
                                             const std::unordered_map<std::string, ov::Layout>& layouts,
                                             const std::function<float(const float)>& transformationFunc,
                                             const std::function<int(const size_t, const int)>& anchorFunc);

std::vector<BoundingBox> parseSSDOutput(const ov::Tensor& tensor, const size_t imgWidth, const size_t imgHeight,
                                        const float confThresh);

void printDetectionBBoxOutputs(const std::vector<BoundingBox>& actualOutput, std::ostringstream& outputStream,
                               const std::vector<std::string>& labels = {});

float boxIntersectionOverUnion(const Box& a, const Box& b);
}  // namespace utils
