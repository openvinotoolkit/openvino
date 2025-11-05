// TODO: Copyright tag

#pragma once

#include <string>
#include <memory>

#include <opencv2/core/mat.hpp>

#include "scenario/accuracy_metrics.hpp"
#include "result.hpp"


class LayerValidator {
public:
    LayerValidator(const std::string& tag, const std::string& layer_name, IAccuracyMetric::Ptr metric);
    Result operator()(const cv::Mat& lhs, const cv::Mat& rhs);

private:
    std::string m_tag;
    std::string m_layer_name;
    IAccuracyMetric::Ptr m_metric;
};
