// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/quantization_granularity_attribute.hpp"

using namespace ov;
using namespace ov;

bool QuantizationGranularityAttribute::is_skipped() const {
    assert((granularity == Granularity::PerChannel) || (granularity == Granularity::PerTensor));
    return granularity != Granularity::PerTensor;
}

std::string QuantizationGranularityAttribute::to_string() const {
    assert((granularity == Granularity::PerChannel) || (granularity == Granularity::PerTensor));

    std::stringstream ss;
    switch (granularity) {
        case Granularity::PerChannel: {
            ss << "PerChannel";
            break;
        }
        case Granularity::PerTensor: {
            ss << "PerTensor";
            break;
        }
        default: {
            ss << "UNKNOWN";
            break;
        }
    }
    return ss.str();
}