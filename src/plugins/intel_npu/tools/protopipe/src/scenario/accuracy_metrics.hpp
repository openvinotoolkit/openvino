//
// Copyright (C) 2023-2024 Intel Corporation
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
