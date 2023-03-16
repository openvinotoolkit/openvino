// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base_segment_precision_validator.hpp"

#include <cmath>

#include "ie_common.h"

namespace ov {
namespace intel_gna {
namespace pass {
namespace pwl {
BaseSegmentPrecisionValidator::BaseSegmentPrecisionValidator(const Function& activation_function,
                                                             double precision,
                                                             double lower_bound,
                                                             double upper_bound)
    : m_precision(precision) {
    m_range = calculate_range(activation_function, lower_bound, upper_bound);
}

bool BaseSegmentPrecisionValidator::is_valid(double error) const {
    return calculate_error_percentage(error, m_range) < m_precision;
}

double BaseSegmentPrecisionValidator::calculate_range(const Function& activation_function,
                                                      double lower_bound,
                                                      double upper_bound) const {
    double delta = (upper_bound - lower_bound) / (kSamplesNum - 1);
    IE_ASSERT(delta > 0.0);

    double min_val = activation_function.get_value(lower_bound);
    double max_val = activation_function.get_value(lower_bound);
    for (int i = 0; i < kSamplesNum; i++) {
        double arg = lower_bound + i * delta;
        double val = activation_function.get_value(arg);
        if (val > max_val)
            max_val = val;
        if (val < min_val)
            min_val = val;
    }
    return std::abs(max_val - min_val);
}

double BaseSegmentPrecisionValidator::calculate_error_percentage(double epsilon, double range) const {
    return (100.0 * std::fabs(epsilon) / range);
}
}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov