// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "configuration_softsign.hpp"

#include "common.hpp"
#include "ie_common.h"
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace intel_gna {
namespace pass {
namespace pwl {

const std::string ConfigurationSoftSign::kActivationName = "Softsign";
constexpr const double ConfigurationSoftSign::kLowerBound;
constexpr const double ConfigurationSoftSign::kUpperBound;
constexpr const double ConfigurationSoftSign::kMinValue;
constexpr const double ConfigurationSoftSign::kMaxValue;
constexpr const double ConfigurationSoftSign::kPerformanceErrorPercentage;
constexpr const double ConfigurationSoftSign::kAccuracyErrorPercentage;
constexpr const double ConfigurationSoftSign::kBreakBound;
constexpr const size_t ConfigurationSoftSign::kMaxIterations;

double ConfigurationSoftSign::FunctionSoftSign::get_value(double x) const {
    return x / (1.0 + std::abs(x));
}

double ConfigurationSoftSign::FunctionSoftSign::get_first_derivative(double x) const {
    return 1.0 / ((1.0 + std::abs(x)) * (1.0 + std::abs(x)));
}

std::shared_ptr<ActivationRepresentationConfiguration> ConfigurationSoftSign::create_config(PWLApproximationMode mode) {
    return std::make_shared<ConfigurationSoftSign>(mode, std::make_shared<BoundariesFQHandlerImpl>());
}

ConfigurationSoftSign::ConfigurationSoftSign(PWLApproximationMode mode,
                                             std::shared_ptr<BoundariesFQHandler> boundaries_handler)
    : ActivationRepresentationConfigurationBase(std::move(boundaries_handler)),
      m_mode(mode) {}

std::shared_ptr<ov::Node> ConfigurationSoftSign::get_pattern_node(std::shared_ptr<ov::Node> input) const {
    return ov::pass::pattern::wrap_type<ov::opset11::SoftSign>({input});
}

std::shared_ptr<ActivationRepresentation> ConfigurationSoftSign::create_representation(
    std::shared_ptr<ov::Node> node,
    std::shared_ptr<ov::Node> fake_quantize) const {
    auto specific_node = std::dynamic_pointer_cast<ov::opset11::SoftSign>(node);
    if (nullptr == specific_node) {
        return nullptr;
    }

    auto surrounding_segments_inserter = std::make_shared<SurroundingSegmentInserterCommon>(kMinValue, kMaxValue);
    FunctionSplitInfoBasedOnBreakBoundValue split_checker(kBreakBound);
    NegationInfoWhenUpperBoundEqualBreakBound negation_checker(kBreakBound);
    SubintervalsCreator creator(split_checker, negation_checker);

    auto data_for_calculation = std::make_shared<ActivationDataForCalculations>(
        std::make_shared<FunctionSoftSign>(),
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