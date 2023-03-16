// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "configuration_exp.hpp"

#include "common.hpp"
#include "ie_common.h"
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace intel_gna {
namespace pass {
namespace pwl {

const std::string ConfigurationExp::kActivationName = "Exp";
const double ConfigurationExp::kLowerBound = -std::log2(std::numeric_limits<int16_t>::max());
const double ConfigurationExp::kUpperBound = std::log10(std::numeric_limits<int16_t>::max());
constexpr const double ConfigurationExp::kMinValue;
constexpr const double ConfigurationExp::kMaxValue;
constexpr const double ConfigurationExp::kPerformanceErrorPercentage;
constexpr const double ConfigurationExp::kAccuracyErrorPercentage;
constexpr const double ConfigurationExp::kBreakBound;
constexpr const size_t ConfigurationExp::kMaxIterations;

double ConfigurationExp::FunctionExp::get_value(double x) const {
    return exp(x);
}

double ConfigurationExp::FunctionExp::get_first_derivative(double x) const {
    return exp(x);
}

std::shared_ptr<ActivationRepresentationConfiguration> ConfigurationExp::create_config(PWLApproximationMode mode) {
    return std::make_shared<ConfigurationExp>(mode, std::make_shared<BoundariesFQHandlerImpl>());
}

ConfigurationExp::ConfigurationExp(PWLApproximationMode mode, std::shared_ptr<BoundariesFQHandler> boundaries_handler)
    : ActivationRepresentationConfigurationBase(std::move(boundaries_handler)),
      m_mode(mode) {}

std::shared_ptr<ov::Node> ConfigurationExp::get_pattern_node(std::shared_ptr<ov::Node> input) const {
    return ov::pass::pattern::wrap_type<ov::opset11::Exp>({input});
}

std::shared_ptr<ActivationRepresentation> ConfigurationExp::create_representation(
    std::shared_ptr<ov::Node> node,
    std::shared_ptr<ov::Node> fake_quantize) const {
    auto specific_node = std::dynamic_pointer_cast<ov::opset11::Exp>(node);
    if (nullptr == specific_node) {
        return nullptr;
    }

    auto surrounding_segments_inserter = std::make_shared<SurroundingSegmentInserterCommon>(kMinValue, kMaxValue);
    FunctionSplitInfoBasedOnBreakBoundValue split_checker(kBreakBound);
    FunctionSubintervalNegationInfoAlwaysTrue negation_checker;
    SubintervalsCreator creator(split_checker, negation_checker);

    auto data_for_calculation = std::make_shared<ActivationDataForCalculations>(
        std::make_shared<FunctionExp>(),
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