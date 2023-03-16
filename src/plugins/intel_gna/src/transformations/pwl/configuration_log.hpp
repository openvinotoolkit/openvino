// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "activation_representation_configuration.hpp"
#include "boundaries_fq_hanlder.hpp"
#include "openvino/runtime/intel_gna/properties.hpp"

namespace ov {
namespace intel_gna {
namespace pass {
namespace pwl {

class ConfigurationLog : public ActivationRepresentationConfigurationBase {
public:
    ConfigurationLog(PWLApproximationMode mode, std::shared_ptr<BoundariesFQHandler> boundaries_handler);

    static std::shared_ptr<ActivationRepresentationConfiguration> create_config(PWLApproximationMode mode);

    std::shared_ptr<ov::Node> get_pattern_node(std::shared_ptr<ov::Node> input) const override;

    std::shared_ptr<ActivationRepresentation> create_representation(
        std::shared_ptr<ov::Node> node,
        std::shared_ptr<ov::Node> fake_quantize) const override;

private:
    class FunctionLog : public Function {
        double get_value(double x) const override;
        double get_first_derivative(double x) const override;
    };

    static const std::string kActivationName;
    static constexpr const double kLowerBound = 0.001;
    static constexpr const double kUpperBound = 2981.0;
    static constexpr const double kMinValue = -11;
    static constexpr const double kMaxValue = std::numeric_limits<int16_t>::max();
    static constexpr const double kPerformanceErrorPercentage = 1.0;
    static constexpr const double kAccuracyErrorPercentage = 0.09;
    static constexpr const double kBreakBound = 0.0;
    static constexpr const size_t kMaxIterations = 5000;

    PWLApproximationMode m_mode;
};
}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov