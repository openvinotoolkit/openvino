// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scenario/accuracy_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include "utils/error.hpp"
#include "utils/logger.hpp"

Norm::Norm(const double tolerance): m_tolerance(tolerance){};

Result Norm::compare(const cv::Mat& lhs, const cv::Mat& rhs) {
    cv::Mat lhsf32, rhsf32;
    lhs.convertTo(lhsf32, CV_32F);
    rhs.convertTo(rhsf32, CV_32F);

    ASSERT(lhsf32.total() == rhsf32.total());
    auto value = cv::norm(lhsf32, rhsf32);
    LOG_DEBUG() << "Norm: " << value << ", tolerance: " << m_tolerance << std::endl;

    // NB: Every comparison against NaN is false, so without this the metric would report a pass.
    if (!std::isfinite(value)) {
        return Error{"Norm is not a finite number - output and/or reference contain NaN/Inf"};
    }

    if (value > m_tolerance) {
        std::stringstream ss;
        ss << value << " > " << m_tolerance;
        return Error{ss.str()};
    }
    return Success{};
}

std::string Norm::str() {
    std::stringstream ss;
    ss << "Norm{tolerance: " << m_tolerance << "}";
    return ss.str();
}

Cosine::Cosine(const double threshold): m_threshold(threshold){};

Result Cosine::compare(const cv::Mat& lhs, const cv::Mat& rhs) {
    cv::Mat lhsf32, rhsf32;
    lhs.convertTo(lhsf32, CV_32F);
    rhs.convertTo(rhsf32, CV_32F);

    ASSERT(lhsf32.total() == rhsf32.total());
    const auto* lhsptr = lhsf32.ptr<float>();
    const auto* rhsptr = rhsf32.ptr<float>();

    double lhsdot = 0.0, rhsdot = 0.0, numr = 0.0;
    for (size_t i = 0; i < lhsf32.total(); ++i) {
        numr += lhsptr[i] * rhsptr[i];
        lhsdot += lhsptr[i] * lhsptr[i];
        rhsdot += rhsptr[i] * rhsptr[i];
    }

    const double eps = 1e-9;
    if (lhsdot < eps || rhsdot < eps) {
        return Error{"Division by zero!"};
    }

    const double similarity = numr / (std::sqrt(lhsdot) * std::sqrt(rhsdot));
    LOG_DEBUG() << "Cosine: " << similarity << ", threshold: " << m_threshold << std::endl;
    // NB: Every comparison against NaN is false, so without this the metric would report a pass.
    if (!std::isfinite(similarity)) {
        return Error{"Cosine similarity is not a finite number - output and/or reference contain NaN/Inf"};
    }
    if (similarity > (1.0 + eps) || similarity < -(1.0 + eps)) {
        std::stringstream ss;
        ss << "Invalid result " << similarity << " (valid range [-1 : +1])";
        return Error{ss.str()};
    }

    if (m_threshold - eps > similarity) {
        std::stringstream ss;
        ss << similarity << " < " << m_threshold;
        return Error{ss.str()};
    }
    return Success{};
}

std::string Cosine::str() {
    std::stringstream ss;
    ss << "Cosine{threshold: " << m_threshold << "}";
    return ss.str();
}

NRMSE::NRMSE(const double tolerance): m_tolerance(tolerance){};

Result NRMSE::compare(const cv::Mat& lhs, const cv::Mat& rhs) {
    cv::Mat lhsf32, rhsf32;
    lhs.convertTo(lhsf32, CV_32F);
    rhs.convertTo(rhsf32, CV_32F);

    const auto size = lhsf32.total();
    if (size == 0) {
        std::stringstream ss;
        ss << "Empty output and reference tensors, nrmse loss set to 0" << std::endl;
        return Success{};
    }

    const auto* lhsptr = lhsf32.ptr<float>();
    const auto* rhsptr = rhsf32.ptr<float>();

    double error = 0.0;
    float lhsmax = 0.0, rhsmax = 0.0, lhsmin = 0.0, rhsmin = 0.0;

    for (size_t i = 0; i < size; ++i) {
        const auto diff = lhsptr[i] - rhsptr[i];
        error += diff * diff;
        lhsmax = std::max(lhsptr[i], lhsmax);
        rhsmax = std::max(rhsptr[i], rhsmax);
        lhsmin = std::min(lhsptr[i], lhsmin);
        rhsmin = std::min(rhsptr[i], rhsmin);
    }

    double nrmse = sqrt(error / size) / std::max(0.001f, std::max(lhsmax - lhsmin, rhsmax - rhsmin));
    LOG_DEBUG() << "NRMSE: " << nrmse << ", tolerance: " << m_tolerance << std::endl;

    // NB: Every comparison against NaN is false, so without this the metric would report a pass.
    if (!std::isfinite(nrmse)) {
        return Error{"NRMSE is not a finite number - output and/or reference contain NaN/Inf"};
    }

    if (m_tolerance < nrmse) {
        std::stringstream ss;
        ss << nrmse << " > " << m_tolerance;
        return Error{ss.str()};
    }
    return Success{};
}

std::string NRMSE::str() {
    std::stringstream ss;
    ss << "nrmse{tolerance: " << m_tolerance << "}";
    return ss.str();
}

namespace {

constexpr double kEps = 1e-9;

struct Detection {
    float x1 = 0.f;
    float y1 = 0.f;
    float x2 = 0.f;
    float y2 = 0.f;
    float score = 0.f;
    int label = 0;
};

float intersectionOverUnion(const Detection& a, const Detection& b) {
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    const float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
    const float area_a = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
    const float area_b = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
    const float uni = area_a + area_b - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

// Drops unit dimensions, e.g. [1, 300, 6] -> [300, 6] and [1, 1, N, 7] -> [N, 7].
std::vector<int> squeezeDims(const cv::Mat& mat) {
    std::vector<int> dims;
    for (int i = 0; i < mat.size.dims(); ++i) {
        if (mat.size[i] != 1) {
            dims.push_back(mat.size[i]);
        }
    }
    while (dims.size() < 2u) {
        dims.insert(dims.begin(), 1);
    }
    return dims;
}

enum class DetectionLayout {
    XYXY,       // [x1, y1, x2, y2, score, label]
    SSD,        // [image_id, label, score, x1, y1, x2, y2]
    RAW,        // [cx, cy, w, h, class scores...]
    RAW_WITH_OBJECTNESS  // [cx, cy, w, h, objectness, class scores...]
};

struct LayoutInfo {
    DetectionLayout layout;
    int num_boxes = 0;
    int stride = 0;
    // Raw detection heads are commonly emitted transposed, as [4 + num_classes, num_boxes].
    bool transposed = false;
};

LayoutInfo resolveLayout(const std::vector<int>& dims, int num_classes) {
    const int rows = dims[0];
    const int cols = dims[1];
    // Raw layouts are probed first, they are only reachable when num_classes is set explicitly.
    if (num_classes > 0) {
        if (cols == num_classes + 4) {
            return {DetectionLayout::RAW, rows, cols, false};
        }
        if (cols == num_classes + 5) {
            return {DetectionLayout::RAW_WITH_OBJECTNESS, rows, cols, false};
        }
        if (rows == num_classes + 4) {
            return {DetectionLayout::RAW, cols, rows, true};
        }
        if (rows == num_classes + 5) {
            return {DetectionLayout::RAW_WITH_OBJECTNESS, cols, rows, true};
        }
    }
    if (cols == 6) {
        return {DetectionLayout::XYXY, rows, cols, false};
    }
    if (cols == 7) {
        return {DetectionLayout::SSD, rows, cols, false};
    }
    THROW_ERROR("MAP metric failed to interpret detection tensor of shape ["
                << rows << ", " << cols << "]. Expected trailing dimension 6 ([x1, y1, x2, y2, score, label]), "
                << "7 ([image_id, label, score, x1, y1, x2, y2]) or, for raw outputs, "
                << "\"num_classes\" + 4 / \"num_classes\" + 5");
}

std::string shapeToString(const cv::Mat& mat) {
    std::stringstream ss;
    ss << "[";
    for (int i = 0; i < mat.size.dims(); ++i) {
        ss << (i != 0 ? ", " : "") << mat.size[i];
    }
    ss << "] (" << mat.total() << " element(s), depth " << mat.depth() << ")";
    return ss.str();
}

std::vector<Detection> decodeDetections(const cv::Mat& mat, const MAP::Params& params) {
    // NB: Guard before convertTo() - an unexpected/corrupted header would otherwise
    // turn straight into a huge allocation instead of a readable error.
    constexpr size_t kMaxDetectionElements = 1ull << 26;
    if (mat.empty() || mat.total() == 0u || (mat.total() > kMaxDetectionElements)) {
        THROW_ERROR("MAP metric got an implausible detection tensor: " << shapeToString(mat));
    }

    cv::Mat f32;
    mat.convertTo(f32, CV_32F);
    if (!f32.isContinuous()) {
        f32 = f32.clone();
    }

    const auto dims = squeezeDims(f32);
    if (dims.size() != 2u) {
        THROW_ERROR("MAP metric expects a detection tensor with two non-unit dimensions, but got " << dims.size());
    }
    const auto info = resolveLayout(dims, params.num_classes);

    const float* ptr = f32.ptr<float>();
    auto at = [ptr, &info](int record, int field) {
        const size_t idx = info.transposed ? static_cast<size_t>(field) * info.num_boxes + record
                                           : static_cast<size_t>(record) * info.stride + field;
        return ptr[idx];
    };

    std::vector<Detection> dets;
    dets.reserve(info.num_boxes);
    for (int i = 0; i < info.num_boxes; ++i) {
        Detection d;
        if (info.layout == DetectionLayout::XYXY) {
            d.x1 = at(i, 0);
            d.y1 = at(i, 1);
            d.x2 = at(i, 2);
            d.y2 = at(i, 3);
            d.score = at(i, 4);
            d.label = static_cast<int>(std::lround(at(i, 5)));
        } else if (info.layout == DetectionLayout::SSD) {
            d.label = static_cast<int>(std::lround(at(i, 1)));
            d.score = at(i, 2);
            d.x1 = at(i, 3);
            d.y1 = at(i, 4);
            d.x2 = at(i, 5);
            d.y2 = at(i, 6);
        } else {
            const bool has_objectness = info.layout == DetectionLayout::RAW_WITH_OBJECTNESS;
            const int scores_offset = has_objectness ? 5 : 4;
            float best_score = 0.f;
            int best_label = -1;
            for (int c = 0; c < params.num_classes; ++c) {
                const float score = at(i, scores_offset + c);
                if (score > best_score) {
                    best_score = score;
                    best_label = c;
                }
            }
            if (best_label < 0) {
                continue;
            }
            d.label = best_label;
            d.score = has_objectness ? best_score * at(i, 4) : best_score;
            // Raw heads encode boxes as center/size.
            const float cx = at(i, 0);
            const float cy = at(i, 1);
            const float w = at(i, 2);
            const float h = at(i, 3);
            d.x1 = cx - w / 2.f;
            d.y1 = cy - h / 2.f;
            d.x2 = cx + w / 2.f;
            d.y2 = cy + h / 2.f;
        }

        // Skip padding slots emitted by fixed-size detection heads.
        if (d.label < 0 || d.score <= 0.f || (d.x2 - d.x1) <= 0.f || (d.y2 - d.y1) <= 0.f) {
            continue;
        }
        if (d.score < params.confidence_threshold) {
            continue;
        }
        dets.push_back(d);
    }
    return dets;
}

// Greedy per-class non-maximum suppression. Returns detections sorted by descending score.
std::vector<Detection> nonMaximumSuppression(std::vector<Detection> dets, double nms_threshold) {
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });

    if (nms_threshold <= 0.0) {
        return dets;
    }
    std::vector<char> suppressed(dets.size(), 0);
    std::vector<Detection> kept;
    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        kept.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j] || dets[j].label != dets[i].label) {
                continue;
            }
            if (intersectionOverUnion(dets[i], dets[j]) > nms_threshold) {
                suppressed[j] = 1;
            }
        }
    }
    return kept;
}

// Average Precision for a single class using all-point (COCO/VOC2010+) interpolation.
// "preds" must be sorted by descending score.
double averagePrecision(const std::vector<Detection>& preds, const std::vector<Detection>& gts, double iou_threshold) {
    const size_t num_gt = gts.size();
    if (num_gt == 0) {
        return preds.empty() ? 1.0 : 0.0;
    }

    std::vector<char> matched(num_gt, 0);
    std::vector<double> recall(preds.size());
    std::vector<double> precision(preds.size());
    double tp = 0.0;
    double fp = 0.0;
    for (size_t i = 0; i < preds.size(); ++i) {
        float best_iou = 0.f;
        int best_j = -1;
        for (size_t j = 0; j < num_gt; ++j) {
            if (matched[j]) {
                continue;
            }
            const float value = intersectionOverUnion(preds[i], gts[j]);
            if (value > best_iou) {
                best_iou = value;
                best_j = static_cast<int>(j);
            }
        }
        if (best_j >= 0 && best_iou >= iou_threshold) {
            matched[best_j] = 1;
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        recall[i] = tp / static_cast<double>(num_gt);
        precision[i] = tp / (tp + fp);
    }

    // Make the precision curve monotonically non-increasing from the right.
    for (int i = static_cast<int>(precision.size()) - 2; i >= 0; --i) {
        precision[i] = std::max(precision[i], precision[i + 1]);
    }
    // Integrate the precision envelope over recall.
    double ap = 0.0;
    double prev_recall = 0.0;
    for (size_t i = 0; i < precision.size(); ++i) {
        ap += (recall[i] - prev_recall) * precision[i];
        prev_recall = recall[i];
    }
    return ap;
}

// Detections are already sorted by descending score at this point.
void logDetections(const char* tag, const std::vector<Detection>& dets) {
    if (Logger::global_lvl < LogLevel::Debug) {
        return;
    }
    constexpr size_t kMaxLogged = 20u;
    const size_t shown = std::min(kMaxLogged, dets.size());
    LOG_DEBUG() << "MAP: " << tag << " detections, showing " << shown << " of " << dets.size()
                << " (box coordinates are in model input space)" << std::endl;
    for (size_t i = 0; i < shown; ++i) {
        const auto& d = dets[i];
        LOG_DEBUG() << "    score: " << d.score << ", class: " << d.label << ", box: [" << d.x1 << ", " << d.y1
                    << ", " << d.x2 << ", " << d.y2 << "], w: " << (d.x2 - d.x1) << ", h: " << (d.y2 - d.y1)
                    << std::endl;
    }
}

struct ClassSamples {
    std::vector<Detection> preds;
    std::vector<Detection> gts;
};

double computeMAP(const std::vector<Detection>& preds, const std::vector<Detection>& gts,
                  const std::vector<double>& iou_thresholds) {
    if (gts.empty()) {
        // Without reference objects a perfect score requires the model to predict nothing.
        LOG_WARN() << "MAP: reference contains no detections"
                   << (preds.empty() ? ", neither does the actual output - the comparison is vacuous" : "")
                   << std::endl;
        return preds.empty() ? 1.0 : 0.0;
    }

    // AP is averaged over the classes present in the reference.
    std::map<int, ClassSamples> per_class;
    for (const auto& gt : gts) {
        per_class[gt.label].gts.push_back(gt);
    }
    for (const auto& pred : preds) {
        auto it = per_class.find(pred.label);
        if (it != per_class.end()) {
            it->second.preds.push_back(pred);
        }
    }

    double total = 0.0;
    for (double iou_threshold : iou_thresholds) {
        double sum_ap = 0.0;
        for (const auto& [label, samples] : per_class) {
            const double ap = averagePrecision(samples.preds, samples.gts, iou_threshold);
            LOG_DEBUG() << "    class " << label << ": AP: " << ap << " (actual: " << samples.preds.size()
                        << ", reference: " << samples.gts.size() << ")" << std::endl;
            sum_ap += ap;
        }
        const double map_at_iou = sum_ap / static_cast<double>(per_class.size());
        LOG_DEBUG() << "  mAP@" << iou_threshold << ": " << map_at_iou << " over " << per_class.size() << " class(es)"
                    << std::endl;
        total += map_at_iou;
    }
    return total / static_cast<double>(iou_thresholds.size());
}

}  // namespace

MAP::MAP(const Params& params): m_params(params){};

Result MAP::compare(const cv::Mat& lhs, const cv::Mat& rhs) {
    // NB: lhs is the actual model output, rhs is the reference used as ground truth.
    LOG_DEBUG() << "MAP: actual " << shapeToString(lhs) << ", reference " << shapeToString(rhs) << std::endl;
    auto preds = nonMaximumSuppression(decodeDetections(lhs, m_params), m_params.nms_threshold);
    auto gts = nonMaximumSuppression(decodeDetections(rhs, m_params), m_params.nms_threshold);
    LOG_DEBUG() << "MAP: decoded " << preds.size() << " actual and " << gts.size()
                << " reference detection(s) (confidence_threshold: " << m_params.confidence_threshold
                << ", nms_threshold: " << m_params.nms_threshold << ")" << std::endl;
    logDetections("actual", preds);
    logDetections("reference", gts);

    std::vector<double> iou_thresholds;
    if (m_params.averaged_iou) {
        // COCO-style mAP@0.5:0.95.
        for (int i = 0; i < 10; ++i) {
            iou_thresholds.push_back(0.5 + 0.05 * i);
        }
    } else {
        iou_thresholds.push_back(m_params.overlap_threshold);
    }

    const double map = computeMAP(preds, gts, iou_thresholds);
    LOG_DEBUG() << "MAP: " << map << ", map_threshold: " << m_params.map_threshold << std::endl;
    if (m_params.map_threshold - kEps > map) {
        std::stringstream ss;
        ss << map << " < " << m_params.map_threshold;
        return Error{ss.str()};
    }
    std::stringstream ss;
    ss << "mAP: " << map;
    return Success{ss.str()};
}

std::string MAP::str() {
    std::stringstream ss;
    ss << "MAP{map_threshold: " << m_params.map_threshold << ", overlap_threshold: ";
    if (m_params.averaged_iou) {
        ss << "0.5:0.95";
    } else {
        ss << m_params.overlap_threshold;
    }
    ss << ", confidence_threshold: " << m_params.confidence_threshold << ", nms_threshold: "
       << m_params.nms_threshold;
    if (m_params.num_classes > 0) {
        ss << ", num_classes: " << m_params.num_classes;
    }
    ss << "}";
    return ss.str();
}
