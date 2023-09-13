// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "low_precision/lpt_visibility.hpp"
#include "low_precision/rt_info/quantization_granularity_attribute.hpp"

namespace ov {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API PortQuantizationGranularityRestriction {
public:
    PortQuantizationGranularityRestriction(const size_t port, QuantizationGranularityAttribute::Granularity granularity) :
        port(port),
        granularity(granularity) {}

    size_t port;
    QuantizationGranularityAttribute::Granularity granularity;
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ov
