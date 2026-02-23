//
// Copyright (C) 2018-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <map>
#include <string>

namespace utils {

using PerLayerValueMap = std::map<std::string, double>;

PerLayerValueMap parsePerLayerValues(const std::string& str, double defaultValue);

double getValueForLayer(const PerLayerValueMap& valueMap, const std::string& layerName);

} // namespace utils

// Each constant matches the default embedded in its DEFINE_string(flag, "value", ...).
namespace metric_defaults {
    constexpr double prob_tolerance       = 1e-4;
    constexpr double raw_tolerance        = 1e-4;
    constexpr double cosim_threshold      = 0.90;
    constexpr double rrmse_loss_threshold = std::numeric_limits<double>::max();
    constexpr double nrmse_loss_threshold = 0.02;
    constexpr double l2norm_threshold     = 1.0;
    constexpr double overlap_threshold    = 0.50;
    constexpr double map_threshold        = 0.50;
    constexpr double confidence_threshold = 1e-4;
    constexpr double box_tolerance        = 1e-4;
    constexpr double psnr_reference       = 30.0;
    constexpr double psnr_tolerance       = 1e-4;
    constexpr double sem_seg_threshold    = 0.98;
} // namespace metric_defaults
