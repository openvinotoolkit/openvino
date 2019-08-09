// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "myriad_mvnc_wraper.h"
#include "myriad_executor.h"

#include <functional>
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <tuple>

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
    const std::vector<std::string>& SupportedMetrics() const;
    const std::vector<std::string>& SupportedConfigKeys() const;
    const std::vector<std::string>& OptimizationCapabilities() const;
    RangeType RangeForAsyncInferRequests(
        const std::map<std::string, std::string>&) const;

    // Destructor
    ~MyriadMetrics() = default;

private:
    // Data section
    std::vector<std::string> _supportedMetrics;
    std::vector<std::string> _supportedConfigKeys;
    std::vector<std::string> _optimizationCapabilities;
    RangeType _rangeForAsyncInferRequests;
    std::map<std::string, std::string> _idToDeviceFullNameMap;
};

}  // namespace MyriadPlugin
}  // namespace vpu
