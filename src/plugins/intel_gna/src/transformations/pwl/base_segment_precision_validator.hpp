// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "function.hpp"

namespace ov {
namespace intel_gna {
namespace pass {
namespace pwl {
class BaseSegmentPrecisionValidator {
public:
    BaseSegmentPrecisionValidator(const Function& activation_function,
                                  double precision,
                                  double lower_bound,
                                  double upper_bound);

    bool is_valid(double error) const;

private:
    double calculate_range(const Function& activation_function, double lower_bound, double upper_bound) const;
    double calculate_error_percentage(double epsilon, double range) const;

    static constexpr const int kSamplesNum = 500;

    double m_range;
    double m_precision;
};

}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov