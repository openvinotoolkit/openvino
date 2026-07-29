//
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "simulation/layer_validator.hpp"

#include "result.hpp"
#include <sstream>

LayerValidator::LayerValidator(const std::string& tag, const std::string& layer_name, IAccuracyMetric::Ptr metric)
        : m_tag(tag), m_layer_name(layer_name), m_metric(metric) {
}

Result LayerValidator::operator()(const cv::Mat& lhs, const cv::Mat& rhs) {
    auto result = m_metric->compare(lhs, rhs);
    if (!result) {
        std::stringstream ss;
        ss << "Model: " << m_tag << ", Layer: " << m_layer_name << ", Metric: " << m_metric->str()
           << ", Reason: " << result.str() << ";";
        return Error{ss.str()};
    }
    return Success{"Passed"};
}
