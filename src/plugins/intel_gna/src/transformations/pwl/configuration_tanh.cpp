// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "configuration_tanh.hpp"

#include "common.hpp"
#include "ie_common.h"
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace intel_gna {
namespace pass {
namespace pwl {

const std::string ConfigurationTanh::kActivationName = "Tanh";
constexpr const double ConfigurationTanh::kLowerBound;
constexpr const double ConfigurationTanh::kUpperBound;
constexpr const double ConfigurationTanh::kMinValue;
constexpr const double ConfigurationTanh::kMaxValue;
constexpr const double ConfigurationTanh::kPerformanceErrorPercentage;
constexpr const double ConfigurationTanh::kAccuracyErrorPercentage;
constexpr const double ConfigurationTanh::kBreakBound;
constexpr const size_t ConfigurationTanh::kMaxIterations;

double ConfigurationTanh::FunctionTanh::get_value(double x) const {
    return tanh(x);
}

double ConfigurationTanh::FunctionTanh::get_first_derivative(double x) const {
    return 1.0 - tanh(x) * tanh(x);
}

std::shared_ptr<ActivationRepresentationConfiguration> ConfigurationTanh::create_config(PWLApproximationMode mode) {
    return std::make_shared<ConfigurationTanh>(mode, std::make_shared<BoundariesFQHandlerImpl>());
}

ConfigurationTanh::ConfigurationTanh(PWLApproximationMode mode, std::shared_ptr<BoundariesFQHandler> boundaries_handler)
    : ActivationRepresentationConfigurationBase(std::move(boundaries_handler)),
      m_mode(mode) {}

std::shared_ptr<ov::Node> ConfigurationTanh::get_pattern_node(std::shared_ptr<ov::Node> input) const {
    return ov::pass::pattern::wrap_type<ov::opset11::Tanh>({input});
}

std::shared_ptr<ActivationRepresentation> ConfigurationTanh::create_representation(
    std::shared_ptr<ov::Node> node,
    std::shared_ptr<ov::Node> fake_quantize) const {
    auto specific_node = std::dynamic_pointer_cast<ov::opset11::Tanh>(node);
    if (nullptr == specific_node) {
        return nullptr;
    }

    auto surrounding_segments_inserter = std::make_shared<SurroundingSegmentInserterCommon>(kMinValue, kMaxValue);
    FunctionSplitInfoBasedOnBreakBoundValue split_checker(kBreakBound);
    NegationInfoWhenUpperBoundEqualBreakBound negation_checker(kBreakBound);
    SubintervalsCreator creator(split_checker, negation_checker);

    auto data_for_calculation = std::make_shared<ActivationDataForCalculations>(
        std::make_shared<FunctionTanh>(),
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