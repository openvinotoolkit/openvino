// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "activation_representation_configuration.hpp"
#include "openvino/runtime/intel_gna/properties.hpp"

namespace ov {
namespace intel_gna {
namespace pass {
namespace pwl {

class ConfigurationPower : public ActivationRepresentationConfigurationBase {
public:
    ConfigurationPower(PWLApproximationMode mode, std::shared_ptr<BoundariesFQHandler> boundaries_handler);

    static std::shared_ptr<ActivationRepresentationConfiguration> create_config(PWLApproximationMode mode);

    std::shared_ptr<ov::Node> get_pattern_node(std::shared_ptr<ov::Node> input) const override;

    std::shared_ptr<ActivationRepresentation> create_representation(
        std::shared_ptr<ov::Node> node,
        std::shared_ptr<ov::Node> fake_quantize) const override;

private:
    class FunctionPower : public Function {
    public:
        FunctionPower(double exponent, double scale, double shift);
        double get_value(double x) const override;
        double get_first_derivative(double x) const override;

    private:
        double m_exponent;
        double m_scale;
        double m_shift;
    };

    std::vector<FunctionSubinterval> generate_subintervals(double exponent,
                                                           std::shared_ptr<ov::Node> fake_quantize) const;

    bool retrieve_parameters_if_node_is_power(std::shared_ptr<ov::Node> node,
                                              double& exponent,
                                              double& scale,
                                              double& shift) const;
    bool retrieve_parameters_if_node_is_power_ie(std::shared_ptr<ov::Node> node,
                                                 double& exponent,
                                                 double& scale,
                                                 double& shift) const;

    static const std::string kActivationName;
    static constexpr const double kUpperBound = 16;
    // to preserve the same accuraccy as in previous implementation.
    static constexpr const double kPerformanceErrorPercentage = 0.015;
    static constexpr const double kAccuracyErrorPercentage = 0.015;
    static constexpr const double kBreakBound = 0;
    static constexpr const size_t kMaxIterations = 2000;

    PWLApproximationMode m_mode;
};

}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov