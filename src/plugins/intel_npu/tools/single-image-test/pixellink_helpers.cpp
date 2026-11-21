// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pixellink_helpers.hpp"

#include "tensor_utils.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <tuple>

namespace {

// Best link probability over the 8 link channels of one head at pixel (idx).
inline float channelMax(const std::vector<float>& buf, int channels, int hw, int idx) {
    float best = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < channels; ++c) {
        best = std::max(best, buf[static_cast<size_t>(c) * hw + idx]);
    }
    return best;
}

// (H, W) boolean mask (as CV_8U 0/255) of "strongly-linked" pixels for one scale.
// Mirrors _pick_link_mask in the Python reference.
cv::Mat pickLinkMask(const pixellink::ScaleData& s, float linkThresh) {
    const int H = s.H;
    const int W = s.W;
    const int HW = H * W;
    cv::Mat mask(H, W, CV_8U);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int idx = y * W + x;
            const float sh = s.scoresH.at<float>(y, x);
            const float sv = s.scoresV.at<float>(y, x);
            const float lhMax = channelMax(s.linksH, 8, HW, idx);
            const float lvMax = channelMax(s.linksV, 8, HW, idx);
            float linkConf = (sh >= sv) ? lhMax : lvMax;
            linkConf = std::max(linkConf, std::min(lhMax, lvMax));
            mask.at<uint8_t>(y, x) = (linkConf > linkThresh) ? 255 : 0;
        }
    }
    return mask;
}

// Fit ONE oriented 4-point polygon to a point cloud (min-area rectangle).
pixellink::Polygon fitRectToVotes(const std::vector<cv::Point2f>& pts) {
    if (pts.size() < 3) {
        return {};
    }
    const cv::RotatedRect rect = cv::minAreaRect(pts);
    cv::Point2f corners[4];
    rect.points(corners);
    return {corners[0], corners[1], corners[2], corners[3]};
}

// Turn every pixel in a component into 4 corner votes and fit a rectangle.
// delta layout is [dx0,dy0,dx1,dy1,dx2,dy2,dx3,dy3] channel-major, grid-cell units.
pixellink::Polygon decodeComponentPolygon(const std::vector<float>& deltas, int H, int W,
                                          const std::vector<cv::Point>& pix,
                                          float strideW, float strideH) {
    const int HW = H * W;
    std::vector<cv::Point2f> allPts;
    allPts.reserve(pix.size() * 4);
    for (int corner = 0; corner < 4; ++corner) {
        const int cx = 2 * corner;
        const int cy = 2 * corner + 1;
        for (const auto& p : pix) {
            const int idx = p.y * W + p.x;
            const float dx = deltas[static_cast<size_t>(cx) * HW + idx];
            const float dy = deltas[static_cast<size_t>(cy) * HW + idx];
            const float px = (static_cast<float>(p.x) + dx) * strideW;
            const float py = (static_cast<float>(p.y) + dy) * strideH;
            allPts.emplace_back(px, py);
        }
    }
    return fitRectToVotes(allPts);
}

// Greedy polygon NMS keyed on IoU, with a deterministic tie break
// (-score, y_min, x_min). Mirrors nms_polygons in the Python reference.
void nmsPolygons(const std::vector<pixellink::Polygon>& boxes, const std::vector<float>& scores,
                 float iouThresh, std::vector<pixellink::Polygon>& keptBoxes,
                 std::vector<float>& keptScores) {
    keptBoxes.clear();
    keptScores.clear();
    if (boxes.empty()) {
        return;
    }

    auto tieKey = [&](size_t i) {
        float yMin = std::numeric_limits<float>::max();
        float xMin = std::numeric_limits<float>::max();
        for (const auto& pt : boxes[i]) {
            yMin = std::min(yMin, pt.y);
            xMin = std::min(xMin, pt.x);
        }
        return std::make_tuple(-scores[i], yMin, xMin);
    };

    std::vector<size_t> order(boxes.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return tieKey(a) < tieKey(b);
    });

    for (size_t idx : order) {
        bool keep = true;
        for (const auto& kb : keptBoxes) {
            if (pixellink::polygonIoU(boxes[idx], kb) > iouThresh) {
                keep = false;
                break;
            }
        }
        if (keep) {
            keptBoxes.push_back(boxes[idx]);
            keptScores.push_back(scores[idx]);
        }
    }
}

}  // namespace

bool pixellink::buildScales(const std::map<std::string, ov::Tensor>& tensors,
                            std::vector<ScaleData>& scales) {
    struct Head {
        int C;
        int H;
        int W;
        std::string name;
        ov::Tensor tensor;
    };
    std::vector<Head> heads;
    for (const auto& [name, tensor] : tensors) {
        const ov::Shape shape = tensor.get_shape();
        if (shape.size() != 4 || shape[0] != 1) {
            std::cout << "pixellink: skipping tensor '" << name << "' with unsupported shape " << shape
                      << std::endl;
            continue;
        }
        heads.push_back({static_cast<int>(shape[1]), static_cast<int>(shape[2]),
                         static_cast<int>(shape[3]), name, tensor});
    }
    if (heads.empty()) {
        return false;
    }

    // Unique (H, W) scales sorted by resolution, largest first (fpn2, fpn3, fpn4).
    std::vector<std::pair<int, int>> shapes;
    for (const auto& h : heads) {
        const std::pair<int, int> hw{h.H, h.W};
        if (std::find(shapes.begin(), shapes.end(), hw) == shapes.end()) {
            shapes.push_back(hw);
        }
    }
    std::sort(shapes.begin(), shapes.end(), [](const auto& a, const auto& b) {
        return a.first > b.first || (a.first == b.first && a.second > b.second);
    });

    scales.assign(shapes.size(), ScaleData{});
    for (size_t i = 0; i < shapes.size(); ++i) {
        scales[i].H = shapes[i].first;
        scales[i].W = shapes[i].second;
    }
    auto scaleIndex = [&](int H, int W) -> int {
        for (size_t i = 0; i < shapes.size(); ++i) {
            if (shapes[i].first == H && shapes[i].second == W) {
                return static_cast<int>(i);
            }
        }
        return -1;
    };

    for (auto& h : heads) {
        const int si = scaleIndex(h.H, h.W);
        if (si < 0) {
            continue;
        }
        ScaleData& sc = scales[si];

        std::string lname = h.name;
        std::transform(lname.begin(), lname.end(), lname.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        const bool isLink = lname.find("link") != std::string::npos;
        const bool hasVert = lname.find("vert") != std::string::npos;
        const bool hasHori = lname.find("hori") != std::string::npos;

        const ov::Tensor fp32 = npu::utils::toFP32(h.tensor);
        const float* buf = fp32.data<const float>();
        std::vector<float> vec(buf, buf + fp32.get_size());

        if (h.C == 1) {
            // No token -> fill horizontal slot first, then vertical.
            const bool toVert = hasVert || (!hasHori && !sc.scoresH.empty());
            cv::Mat m = cv::Mat(h.H, h.W, CV_32F, const_cast<float*>(vec.data())).clone();
            (toVert ? sc.scoresV : sc.scoresH) = m;
        } else if (h.C == 8 && isLink) {
            const bool toVert = hasVert || (!hasHori && !sc.linksH.empty());
            (toVert ? sc.linksV : sc.linksH) = std::move(vec);
        } else if (h.C == 8) {
            const bool toVert = hasVert || (!hasHori && !sc.deltasH.empty());
            (toVert ? sc.deltasV : sc.deltasH) = std::move(vec);
        }
    }

    for (const auto& sc : scales) {
        const size_t need = static_cast<size_t>(8) * sc.H * sc.W;
        if (sc.scoresH.empty() || sc.scoresV.empty() || sc.deltasH.size() != need ||
            sc.deltasV.size() != need || sc.linksH.size() != need || sc.linksV.size() != need) {
            return false;
        }
    }
    return true;
}

double pixellink::polygonIoU(const Polygon& a, const Polygon& b) {
    if (a.size() < 3 || b.size() < 3) {
        return 0.0;
    }
    const double area1 = std::abs(cv::contourArea(a));
    const double area2 = std::abs(cv::contourArea(b));
    if (area1 <= 0.0 || area2 <= 0.0) {
        return 0.0;
    }
    double interArea = 0.0;
    try {
        std::vector<cv::Point2f> interPoly;
        interArea = cv::intersectConvexConvex(a, b, interPoly, true);
    } catch (const cv::Exception&) {
        // Any degeneracy (colinear points, zero area, etc.) -> treat as no overlap.
        return 0.0;
    }
    const double unionArea = area1 + area2 - interArea;
    return unionArea > 0.0 ? interArea / unionArea : 0.0;
}

void pixellink::decodeLinkAware(const std::vector<ScaleData>& scales, const DecodeParams& params,
                                std::vector<Polygon>& outBoxes, std::vector<float>& outScores) {
    std::vector<Polygon> allBoxes;
    std::vector<float> allScores;

    for (const auto& s : scales) {
        const int H = s.H;
        const int W = s.W;
        if (H <= 0 || W <= 0) {
            continue;
        }
        const float strideH = static_cast<float>(params.origH) / static_cast<float>(H);
        const float strideW = static_cast<float>(params.origW) / static_cast<float>(W);

        const cv::Mat linkMask = pickLinkMask(s, params.linkThresh);

        // Determinism patch 1: round the text score to 3 decimals before thresholding.
        cv::Mat strong(H, W, CV_8U);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const float sh = s.scoresH.at<float>(y, x);
                const float sv = s.scoresV.at<float>(y, x);
                float combined = std::max(sh, sv);
                combined = std::round(combined * 1000.0f) / 1000.0f;
                const bool textOn = combined > params.textThresh;
                const bool linkOn = linkMask.at<uint8_t>(y, x) != 0;
                strong.at<uint8_t>(y, x) = (textOn && linkOn) ? 255 : 0;
            }
        }

        if (params.morphClose) {
            const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::morphologyEx(strong, strong, cv::MORPH_CLOSE, kernel);
        }

        cv::Mat labels;
        const int numLabels = cv::connectedComponents(strong, labels, 8, CV_32S);

        // Single pass to bucket pixels per component (label 0 is background).
        std::vector<std::vector<cv::Point>> compPixels(std::max(numLabels, 1));
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const int lab = labels.at<int>(y, x);
                if (lab > 0) {
                    compPixels[lab].emplace_back(x, y);
                }
            }
        }

        // Determinism patch 2: iterate components in centroid order.
        std::vector<int> ids;
        std::vector<std::pair<double, double>> centroids(numLabels, {0.0, 0.0});
        for (int lab = 1; lab < numLabels; ++lab) {
            const auto& pix = compPixels[lab];
            if (pix.empty()) {
                continue;
            }
            double sy = 0.0;
            double sx = 0.0;
            for (const auto& p : pix) {
                sy += p.y;
                sx += p.x;
            }
            centroids[lab] = {sy / pix.size(), sx / pix.size()};
            ids.push_back(lab);
        }
        std::sort(ids.begin(), ids.end(), [&](int a, int b) {
            return centroids[a] < centroids[b];
        });

        for (int lab : ids) {
            const auto& pix = compPixels[lab];
            if (pix.size() < 2) {
                continue;
            }
            double meanH = 0.0;
            double meanV = 0.0;
            for (const auto& p : pix) {
                meanH += s.scoresH.at<float>(p.y, p.x);
                meanV += s.scoresV.at<float>(p.y, p.x);
            }
            meanH /= pix.size();
            meanV /= pix.size();
            const std::vector<float>& deltaMap = (meanH >= meanV) ? s.deltasH : s.deltasV;
            const float compScore = static_cast<float>(std::max(meanH, meanV));

            const Polygon box = decodeComponentPolygon(deltaMap, H, W, pix, strideW, strideH);
            if (box.size() == 4) {
                allBoxes.push_back(box);
                allScores.push_back(compScore);
            }
        }
    }

    nmsPolygons(allBoxes, allScores, params.nmsIoU, outBoxes, outScores);
}
