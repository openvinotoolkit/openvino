//
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include <opencv2/core/mat.hpp>

#include "scenario/accuracy_metrics.hpp"

class LayerValidator {
public:
    LayerValidator(const std::string& tag, const std::string& layer_name, IAccuracyMetric::Ptr metric);
    Result operator()(const cv::Mat& lhs, const cv::Mat& rhs);

private:
    std::string m_tag;
    std::string m_layer_name;
    IAccuracyMetric::Ptr m_metric;
};
