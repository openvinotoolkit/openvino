// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/common/dequantization_info.hpp"

#include <details/ie_cnn_network_tools.h>
#include <ie_common.h>

#include <blob_factory.hpp>
#include <cmath>
#include <details/caseless.hpp>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cnn_network_impl.hpp"
#include "ie_util_internal.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

DequantizationInfo::DequantizationInfo(const size_t levels, const std::vector<float>& outputLowValues,
                                       const std::vector<float>& outputHighValues)
    : levels(levels), outputLowValues(outputLowValues), outputHighValues(outputHighValues) {
    if (outputLowValues.size() != outputHighValues.size()) {
        THROW_IE_EXCEPTION << "values size is not correct";
    }
}

size_t DequantizationInfo::outputChannels() const {
    return outputHighValues.size();
}
