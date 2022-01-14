// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "myriad_mvnc_wrapper.h"
#include "myriad_executor.h"

#include <functional>
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <tuple>
#include <unordered_set>

using RangeType = std::tuple<unsigned int, unsigned int, unsigned int>;

namespace vpu {
namespace MyriadPlugin {

//------------------------------------------------------------------------------
// class MyriadMetrics
// Class to keep and extract metrics value.
//------------------------------------------------------------------------------

class MyriadMetrics {
public:
    // Constructor
    MyriadMetrics();

    // Accessors
    std::vector<std::string> AvailableDevicesNames(
        const std::shared_ptr<IMvnc> &mvnc,
        const std::vector<DevicePtr> &devicePool) const;

    std::string FullName(std::string deviceName) const;
    std::string DeviceArchitecture(const std::map<std::string, InferenceEngine::Parameter> & options) const;
    float DevicesThermal(const DevicePtr& device) const;
    const std::unordered_set<std::string>& SupportedMetrics() const;
    const std::unordered_set<std::string>& SupportedConfigKeys() const;
    const std::unordered_set<std::string>& OptimizationCapabilities() const;
    RangeType RangeForAsyncInferRequests(
        const std::map<std::string, std::string>&) const;

    // Destructor
    ~MyriadMetrics() = default;

private:
    // Data section
    std::unordered_set<std::string> _supportedMetrics;
    std::unordered_set<std::string> _supportedConfigKeys;
    std::unordered_set<std::string> _optimizationCapabilities;
    RangeType _rangeForAsyncInferRequests;
    std::map<std::string, std::string> _idToDeviceFullNameMap;
};

}  // namespace MyriadPlugin
}  // namespace vpu
