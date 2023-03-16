// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "configuration_sigmoid.hpp"

#include "common.hpp"
#include "ie_common.h"
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace intel_gna {
namespace pass {
namespace pwl {

const std::string ConfigurationSigmoid::kActivationName = "Sigmoid";
constexpr const double ConfigurationSigmoid::kLowerBound;
constexpr const double ConfigurationSigmoid::kUpperBound;
constexpr const double ConfigurationSigmoid::kMinValue;
constexpr const double ConfigurationSigmoid::kMaxValue;
constexpr const double ConfigurationSigmoid::kPerformanceErrorPercentage;
constexpr const double ConfigurationSigmoid::kAccuracyErrorPercentage;
constexpr const double ConfigurationSigmoid::kBreakBound;
constexpr const size_t ConfigurationSigmoid::kMaxIterations;

double ConfigurationSigmoid::FunctionSigmoid::get_value(double x) const {
    return 0.5 * (1.0 + tanh(x / 2.0));
}

double ConfigurationSigmoid::FunctionSigmoid::get_first_derivative(double x) const {
    return get_value(x) * (1.0 - get_value(x));
}

std::shared_ptr<ActivationRepresentationConfiguration> ConfigurationSigmoid::create_config(PWLApproximationMode mode) {
    return std::make_shared<ConfigurationSigmoid>(mode, std::make_shared<BoundariesFQHandlerImpl>());
}

ConfigurationSigmoid::ConfigurationSigmoid(PWLApproximationMode mode,
                                           std::shared_ptr<BoundariesFQHandler> boundaries_handler)
    : ActivationRepresentationConfigurationBase(std::move(boundaries_handler)),
      m_mode(mode) {}

std::shared_ptr<ov::Node> ConfigurationSigmoid::get_pattern_node(std::shared_ptr<ov::Node> input) const {
    return ov::pass::pattern::wrap_type<ov::opset11::Sigmoid>({input});
}

std::shared_ptr<ActivationRepresentation> ConfigurationSigmoid::create_representation(
    std::shared_ptr<ov::Node> node,
    std::shared_ptr<ov::Node> fake_quantize) const {
    auto specific_node = std::dynamic_pointer_cast<ov::opset11::Sigmoid>(node);
    if (nullptr == specific_node) {
        return nullptr;
    }

    auto surrounding_segments_inserter = std::make_shared<SurroundingSegmentInserterCommon>(kMinValue, kMaxValue);
    FunctionSplitInfoBasedOnBreakBoundValue split_checker(kBreakBound);
    NegationInfoWhenUpperBoundEqualBreakBound negation_checker(kBreakBound);
    SubintervalsCreator creator(split_checker, negation_checker);

    auto data_for_calculation = std::make_shared<ActivationDataForCalculations>(
        std::make_shared<FunctionSigmoid>(),
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