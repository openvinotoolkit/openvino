// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/runtime/tensor.hpp>

// PixelLink + EAST hybrid post-processing (pso0-style scene-text detector).
// Ports the link-aware decoder and the COCO-style oriented-polygon mAP from
// the standalone Python reference into the single-image-test tool. All polygon
// geometry goes through OpenCV (no shapely / no external deps).
namespace pixellink {

// An oriented quadrilateral in image space (4 corners, x/y order).
using Polygon = std::vector<cv::Point2f>;

// One FPN scale's six PixelLink heads. The 8-channel buffers are stored flat
// in channel-major order: buf[c * H * W + y * W + x].
struct ScaleData {
    int H = 0;
    int W = 0;
    cv::Mat scoresH;             // CV_32F, H x W  (text score, horizontal head)
    cv::Mat scoresV;             // CV_32F, H x W  (text score, vertical head)
    std::vector<float> deltasH;  // 8 * H * W      (4 corner (dx,dy), horiz head)
    std::vector<float> deltasV;  // 8 * H * W      (4 corner (dx,dy), vert head)
    std::vector<float> linksH;   // 8 * H * W      (8 link directions, horiz head)
    std::vector<float> linksV;   // 8 * H * W      (8 link directions, vert head)
};

struct DecodeParams {
    float textThresh = 0.55f;
    float linkThresh = 0.60f;
    float nmsIoU = 0.50f;       // cross-scale NMS IoU inside the decoder
    bool morphClose = false;
    int origH = 768;            // image space the grid strides map back to
    int origW = 1152;
};

// Map the model's output tensors to per-scale PixelLink heads. Groups 4D
// [1, C, H, W] tensors by spatial (H, W) into FPN scales (largest first) and
// classifies each head by name tokens: "link" -> link head, "score" -> score
// head, otherwise delta/bbox head; "vert"/"hori" select the vertical/horizontal
// head. Returns false if any scale is missing one of its six heads.
bool buildScales(const std::map<std::string, ov::Tensor>& tensors, std::vector<ScaleData>& scales);

// IoU between two oriented convex quadrilaterals (OpenCV intersectConvexConvex).
double polygonIoU(const Polygon& a, const Polygon& b);

// Link-aware PixelLink decoder: text mask AND link mask -> connected components
// -> one min-area rectangle per component (fit over the per-pixel corner votes),
// then greedy cross-scale IoU NMS. Deterministic (score rounding + centroid order).
void decodeLinkAware(const std::vector<ScaleData>& scales,
                     const DecodeParams& params,
                     std::vector<Polygon>& outBoxes,
                     std::vector<float>& outScores);

}  // namespace pixellink
