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

class ConfigurationExp : public ActivationRepresentationConfigurationBase {
public:
    static std::shared_ptr<ActivationRepresentationConfiguration> create_config(PWLApproximationMode mode);

    ConfigurationExp(PWLApproximationMode mode, std::shared_ptr<BoundariesFQHandler> boundaries_handler);

    std::shared_ptr<ov::Node> get_pattern_node(std::shared_ptr<ov::Node> input) const override;

    std::shared_ptr<ActivationRepresentation> create_representation(
        std::shared_ptr<ov::Node> node,
        std::shared_ptr<ov::Node> fake_quantize) const override;

private:
    class FunctionExp : public Function {
        double get_value(double x) const override;
        double get_first_derivative(double x) const override;
    };

    static const std::string kActivationName;
    static const double kLowerBound;
    static const double kUpperBound;
    static constexpr const double kMinValue = 0;
    static constexpr const double kMaxValue = std::numeric_limits<int16_t>::max();
    static constexpr const double kPerformanceErrorPercentage = 1.0;
    static constexpr const double kAccuracyErrorPercentage = 0.09;
    static constexpr const double kBreakBound = 0.045;
    static constexpr const size_t kMaxIterations = 2000;

    PWLApproximationMode m_mode;
};

}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov