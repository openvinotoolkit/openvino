// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <opencv2/core/core.hpp>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "result.hpp"

struct IAccuracyMetric {
    using Ptr = std::shared_ptr<IAccuracyMetric>;
    virtual Result compare(const cv::Mat& lhs, const cv::Mat& rhs) = 0;
    // Called at build time with the shape of every output layer this metric validates.
    virtual void bindLayer(const std::vector<int>& /*dims*/) {}
    virtual std::string str() = 0;
    virtual ~IAccuracyMetric() = default;
};

class Norm : public IAccuracyMetric {
public:
    using Ptr = std::shared_ptr<Norm>;
    explicit Norm(const double tolerance);
    Result compare(const cv::Mat& lhs, const cv::Mat& rhs) override;
    std::string str() override;

private:
    double m_tolerance;
};

class Cosine : public IAccuracyMetric {
public:
    using Ptr = std::shared_ptr<Cosine>;
    explicit Cosine(const double threshold);
    Result compare(const cv::Mat& lhs, const cv::Mat& rhs) override;
    std::string str() override;

private:
    double m_threshold;
};

class NRMSE : public IAccuracyMetric {
public:
    using Ptr = std::shared_ptr<NRMSE>;
    explicit NRMSE(const double tolerance);
    Result compare(const cv::Mat& lhs, const cv::Mat& rhs) override;
    std::string str() override;

private:
    double m_tolerance;
};

class MAP : public IAccuracyMetric {
public:
    using Ptr = std::shared_ptr<MAP>;

    // Encoding of DETR-style [num_queries, 4] box outputs.
    enum class BoxFormat { CXCYWH, XYXY };

    struct Params {
        // Minimal mAP value to treat the comparison as passed.
        double map_threshold = 0.5;
        // IoU used to match a prediction against a reference box. Ignored when averaged_iou is set.
        double overlap_threshold = 0.5;
        // mAP@0.5:0.95 - average the result over IoU thresholds [0.5, 0.95] with step 0.05.
        bool averaged_iou = false;
        double confidence_threshold = 0.0;
        double nms_threshold = 0.45;
        // Required only to decode raw (not post-processed) detection outputs.
        int num_classes = -1;
        // Box encoding of DETR-style [N, 4] outputs, detected from the data when unset.
        std::optional<BoxFormat> box_format;
    };

    explicit MAP(const Params& params);
    Result compare(const cv::Mat& lhs, const cv::Mat& rhs) override;
    void bindLayer(const std::vector<int>& dims) override;
    std::string str() override;

private:
    Result evaluate(const cv::Mat& lhs, const cv::Mat& rhs);

    Params m_params;
    std::set<std::vector<int>> m_bound_dims;
    // Squeezed {boxes, logits} shapes, set only when the bound layers form a DETR pair.
    std::optional<std::pair<std::vector<int>, std::vector<int>>> m_detr_dims;
    // DETR layer waiting for its companion, as {actual, reference}.
    std::optional<std::pair<cv::Mat, cv::Mat>> m_detr_boxes;
    std::optional<std::pair<cv::Mat, cv::Mat>> m_detr_logits;
};
