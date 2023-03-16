// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "configuration_power.hpp"

#include "common.hpp"
#include "common/graph_utils.hpp"
#include "common/numerical_utils.hpp"
#include "ie_common.h"
#include "legacy/ngraph_ops/power.hpp"
#include "log/debug.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace intel_gna {
using namespace common;
namespace pass {
namespace pwl {

const std::string ConfigurationPower::kActivationName = "Exp";
constexpr const double ConfigurationPower::kUpperBound;
constexpr const double ConfigurationPower::kPerformanceErrorPercentage;
constexpr const double ConfigurationPower::kAccuracyErrorPercentage;
constexpr const double ConfigurationPower::kBreakBound;
constexpr const size_t ConfigurationPower::kMaxIterations;

ConfigurationPower::FunctionPower::FunctionPower(double exponent, double scale, double shift)
    : m_exponent(exponent),
      m_scale(scale),
      m_shift(shift) {}

double ConfigurationPower::FunctionPower::get_value(double x) const {
    return pow(x * m_scale + m_shift, m_exponent);
}

double ConfigurationPower::FunctionPower::get_first_derivative(double x) const {
    return m_exponent * m_scale * pow(m_shift + x * m_scale, m_exponent - 1);
}

ConfigurationPower::ConfigurationPower(PWLApproximationMode mode,
                                       std::shared_ptr<BoundariesFQHandler> boundaries_handler)
    : ActivationRepresentationConfigurationBase(std::move(boundaries_handler)),
      m_mode(mode) {}

std::shared_ptr<ActivationRepresentationConfiguration> ConfigurationPower::create_config(PWLApproximationMode mode) {
    return std::make_shared<ConfigurationPower>(mode, std::make_shared<BoundariesFQHandlerImpl>(true));
}

std::shared_ptr<ov::Node> ConfigurationPower::get_pattern_node(std::shared_ptr<ov::Node> input) const {
    auto power_default = ov::pass::pattern::wrap_type<ov::opset11::Power>({input, ov::pass::pattern::any_input()});
    auto power_with_defined_broadcast = ov::pass::pattern::wrap_type<ov::opset11::Power>(
        {input, ov::pass::pattern::any_input(), ov::pass::pattern::any_input()});
    auto power_ie = ov::pass::pattern::wrap_type<ngraph::op::PowerIE>({input});
    return std::make_shared<ov::pass::pattern::op::Or>(
        ov::OutputVector{power_default, power_with_defined_broadcast, power_ie});
}

std::shared_ptr<ActivationRepresentation> ConfigurationPower::create_representation(
    std::shared_ptr<ov::Node> node,
    std::shared_ptr<ov::Node> fake_quantize) const {
    double exponent = 0;
    double scale = 0;
    double shift = 0;

    if (!retrieve_parameters_if_node_is_power(node, exponent, scale, shift) &&
        !retrieve_parameters_if_node_is_power_ie(node, exponent, scale, shift)) {
        return nullptr;
    }

    if (AreFpEq(exponent, 1.0)) {
        // An affine primitive will be used in this case.
        return nullptr;
    }

    if (AreFpEq(exponent, 0.0)) {
        std::vector<Segment> pwls;
        pwls.emplace_back(0, 1, -std::numeric_limits<double>::infinity());
        pwls.emplace_back(0, 1, std::numeric_limits<double>::infinity());
        pwls.emplace_back(0, 0, std::numeric_limits<double>::infinity());

        auto precalculated_segments = std::make_shared<PrecalculatedActivationSegments>(pwls);
        return std::make_shared<ActivationRepresentationImpl>(kActivationName, precalculated_segments);
    }

    auto surrounding_segments_inserter = std::make_shared<SurroundingSegmentInserterPower>(exponent);

    auto data_for_calculation = std::make_shared<ActivationDataForCalculations>(
        std::make_shared<FunctionPower>(exponent, scale, shift),
        generate_subintervals(exponent, fake_quantize),
        surrounding_segments_inserter,
        get_allowed_error_percentage(m_mode, kAccuracyErrorPercentage, kPerformanceErrorPercentage),
        kMaxIterations);

    return std::make_shared<ActivationRepresentationImpl>(kActivationName, data_for_calculation);
}

std::vector<FunctionSubinterval> ConfigurationPower::generate_subintervals(
    double exponent,
    std::shared_ptr<ov::Node> fake_quantize) const {
    auto lower_bound = common::AreFpEq(fmod(exponent, 1.0), 0.0) ? -16.0 : 0.0;

    FunctionSplitInfoBasedOnBreakBoundValue split_checker(kBreakBound);
    FunctionSubintervalNegationInfoPower negation_checker(exponent, kBreakBound);
    SubintervalsCreator creator(split_checker, negation_checker);

    return ActivationRepresentationConfigurationBase::generate_subintervals(fake_quantize,
                                                                            creator,
                                                                            lower_bound,
                                                                            kUpperBound);
}

bool ConfigurationPower::retrieve_parameters_if_node_is_power(std::shared_ptr<ov::Node> node,
                                                              double& exponent,
                                                              double& scale,
                                                              double& shift) const {
    auto specific_node = std::dynamic_pointer_cast<ov::opset11::Power>(node);
    if (nullptr == specific_node) {
        return false;
    }

    auto constant = std::dynamic_pointer_cast<ov::opset11::Constant>(node->get_input_node_shared_ptr(1));
    if (!graph_utils::get_constant_value(constant, exponent)) {
        THROW_GNA_EXCEPTION << "The unsupported type of element.";
    }

    scale = 1;
    shift = 0;

    return true;
}

bool ConfigurationPower::retrieve_parameters_if_node_is_power_ie(std::shared_ptr<ov::Node> node,
                                                                 double& exponent,
                                                                 double& scale,
                                                                 double& shift) const {
    auto specific_node = std::dynamic_pointer_cast<ngraph::op::PowerIE>(node);

    if (nullptr == specific_node) {
        return false;
    }

    exponent = specific_node->power;
    scale = specific_node->scale;
    shift = specific_node->shift;

    return true;
}

}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov