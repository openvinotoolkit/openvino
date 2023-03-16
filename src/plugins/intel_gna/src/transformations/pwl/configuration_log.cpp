// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "configuration_log.hpp"

#include "common.hpp"
#include "ie_common.h"
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace intel_gna {
namespace pass {
namespace pwl {

const std::string ConfigurationLog::kActivationName = "Log";
constexpr const double ConfigurationLog::kLowerBound;
constexpr const double ConfigurationLog::kUpperBound;
constexpr const double ConfigurationLog::kMinValue;
constexpr const double ConfigurationLog::kMaxValue;
constexpr const double ConfigurationLog::kPerformanceErrorPercentage;
constexpr const double ConfigurationLog::kAccuracyErrorPercentage;
constexpr const double ConfigurationLog::kBreakBound;
constexpr const size_t ConfigurationLog::kMaxIterations;

double ConfigurationLog::FunctionLog::get_value(double x) const {
    return std::log(x);
}

double ConfigurationLog::FunctionLog::get_first_derivative(double x) const {
    return 1.0 / x;
}

std::shared_ptr<ActivationRepresentationConfiguration> ConfigurationLog::create_config(PWLApproximationMode mode) {
    // prvious implementation didn't used does not use fake_quantize at all for Log
    return std::make_shared<ConfigurationLog>(mode, std::make_shared<BoundariesFQHandler>());
}

ConfigurationLog::ConfigurationLog(PWLApproximationMode mode, std::shared_ptr<BoundariesFQHandler> boundaries_handler)
    : ActivationRepresentationConfigurationBase(std::move(boundaries_handler)),
      m_mode(mode) {}

std::shared_ptr<ov::Node> ConfigurationLog::get_pattern_node(std::shared_ptr<ov::Node> input) const {
    return ov::pass::pattern::wrap_type<ov::opset11::Log>({input});
}

std::shared_ptr<ActivationRepresentation> ConfigurationLog::create_representation(
    std::shared_ptr<ov::Node> node,
    std::shared_ptr<ov::Node> fake_quantize) const {
    auto specific_node = std::dynamic_pointer_cast<ov::opset11::Log>(node);
    if (nullptr == specific_node) {
        return nullptr;
    }

    auto surrounding_segments_inserter = std::make_shared<SurroundingSegmentInserterCommon>(kMinValue, kMaxValue);
    FunctionSplitInfoNever split_checker;
    FunctionSubintervalNegationInfoAlwaysFalse negation_checker;
    SubintervalsCreator creator(split_checker, negation_checker);

    auto data_for_calculation = std::make_shared<ActivationDataForCalculations>(
        std::make_shared<FunctionLog>(),
        generate_subintervals(fake_quantize, creator, kLowerBound, kUpperBound),
        surrounding_segments_inserter,
        get_allowed_error_percentage(m_mode, kAccuracyErrorPercentage, kPerformanceErrorPercentage),
        kMaxIterations);

    return std::make_shared<ActivationRepresentationImpl>(kActivationName, data_for_calculation);
}

}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov