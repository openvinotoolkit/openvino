// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/intel_gna/properties.hpp"

namespace ov {
namespace intel_gna {
namespace pass {
namespace pwl {

double get_allowed_error_percentage(const PWLApproximationMode& mode,
                                    double accuracy_precision_in_precent,
                                    double performance_precision_in_percent);

}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov