// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "myriad_config.h"
#include "myriad_mvnc_wraper.h"

#include <memory>
#include <string>
#include <vector>

namespace vpu {
namespace MyriadPlugin {

struct DeviceDesc {
    int _graphNum = 0;
    int _maxGraphNum = 0;
    std::string _name;
    ncDevicePlatform_t _platform = NC_ANY_PLATFORM;
    ncDeviceProtocol_t _protocol = NC_ANY_PROTOCOL;

    int _deviceIdx = -1;
    ncDeviceHandle_t *_deviceHandle = nullptr;

    bool isBooted() const {
        return _deviceHandle != nullptr;
    }
    bool isEmpty() const {
        return _graphNum == 0;
    }
    bool isNotFull() const {
        return _graphNum < _maxGraphNum;
    }

    bool isSuitableForConfig(const MyriadConfig& config) const {
        bool isSuitableByName = true;
        if (!config.deviceName().empty()) {
            isSuitableByName = config.deviceName() == _name;
        }

        return isSuitableByName &&
               ((config.platform() == NC_ANY_PLATFORM) || (_platform == config.platform())) &&
               ((config.protocol() == NC_ANY_PROTOCOL) || (_protocol == config.protocol()));
    }

    Platform revision() const {
        VPU_THROW_UNLESS(_platform != NC_ANY_PLATFORM, "Cannot get a revision from not booted device");
        return _platform == NC_MYRIAD_2 ? Platform::MYRIAD_2 : Platform::MYRIAD_X;
    }
};

typedef std::shared_ptr<DeviceDesc> DevicePtr;

class MyriadDeviceProvider {
public:
    MyriadDeviceProvider(bool forceReset, std::shared_ptr<IMvnc> mvnc, const Logger::Ptr& log);
    ~MyriadDeviceProvider() = default;

    /**
     * @brief Get myriad device
     * @return Already booted and empty device or new booted device
     */
    DevicePtr openDevice(std::vector<DevicePtr> &devicePool, const MyriadConfig& config);

    static void closeDevices(std::vector<DevicePtr> &devicePool, std::shared_ptr<IMvnc> mvnc);

private:
    /**
     * @brief Try to boot any available device that suitable for selected platform and protocol
     * @param configPlatform Boot the selected platform
     * @param configProtocol Boot device with selected protocol
     */
    ncStatus_t bootNextDevice(std::vector<DevicePtr> &devicePool,
                              const MyriadConfig& config);

private:
    Logger::Ptr _log;
    std::shared_ptr<IMvnc> _mvnc;
};

using MyriadDeviceProviderPtr = std::unique_ptr<MyriadDeviceProvider>;

}  // namespace MyriadPlugin
}  // namespace vpu
