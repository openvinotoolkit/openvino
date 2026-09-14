//
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <opencv2/core/core.hpp>
#include <string>

#include "result.hpp"

struct IAccuracyMetric {
    using Ptr = std::shared_ptr<IAccuracyMetric>;
    virtual Result compare(const cv::Mat& lhs, const cv::Mat& rhs) = 0;
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
    };

    explicit MAP(const Params& params);
    Result compare(const cv::Mat& lhs, const cv::Mat& rhs) override;
    std::string str() override;

private:
    Params m_params;
};
